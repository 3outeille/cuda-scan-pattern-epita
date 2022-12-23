#include "to_bench.cuh"

#include "cuda_tools/cuda_error_checking.cuh"
#include "cuda_tools/host_shared_ptr.cuh"

#include <cuda_profiler_api.h>
#include <iostream>

template <typename T>
static __global__ void single_pass_chained_scan(T *buffer, T *reduce_arr, int buffer_size, int reduce_arr_size)
{
    int tid = threadIdx.x;
    int id = threadIdx.x + (blockIdx.x * blockDim.x);

    __shared__ int shared_memory[1024];

    if (id < buffer_size)
        shared_memory[tid] = buffer[id];
    else
        shared_memory[tid] = 0;

    __syncthreads();

    // Reduce
    for (int stride = 1; stride < blockDim.x; stride *= 2)
    {
        if (tid % (2 * stride) == 0)
            shared_memory[tid] += shared_memory[tid + stride];
        __syncthreads();
    }

    if (tid == 0)
        reduce_arr[blockIdx.x] = shared_memory[0];

    __syncthreads();

    // Adjacent synchronization
    if (tid == 0)
    {
        if (blockIdx.x == 0)
            reduce_arr[0] = shared_memory[0];
        else
        {
            // Force reduce_arr to be in global memory (and not register) 
            while(atomicAdd(&reduce_arr[blockIdx.x-1], 0) == 0){
                // Wait until previous block finish to compute its reduce.
            }
            reduce_arr[blockIdx.x] = reduce_arr[blockIdx.x - 1] + shared_memory[0];
        }
        
        printf("(id-sync: %d): %d\n", blockIdx.x, reduce_arr[blockIdx.x]);

        // FIXME: Add reduction value to 1st element of block
        if (blockIdx.x > 0) {
            int prev_prefix = (blockIdx.x >= 3) ? reduce_arr[blockIdx.x - 2] : 0;
            atomicAdd(&buffer[blockIdx.x * blockDim.x], reduce_arr[blockIdx.x - 1] + prev_prefix);
        }
    }

    __syncthreads();

    // Scan
    int x;

    for (int stride = 1; stride < blockDim.x; stride *= 2)
    {
        if (tid >= stride && (id % blockDim.x) != 0)
            x = buffer[id] + buffer[id - stride];

        __syncthreads();

        if (tid >= stride && (id % blockDim.x) != 0)
            buffer[id] = x;

        __syncthreads();
    }
}

void scan_opti_4(cuda_tools::host_shared_ptr<int> buffer)
{
    // Reduce-then-scan (reduce-scan-scan)
    cudaProfilerStart();

    constexpr int nb_threads = 2;
    const int nb_blocks = (buffer.size_ + nb_threads - 1) / nb_threads;
    std::cout << "nb_blocks: " << nb_blocks << std::endl;
    std::cout << "nb_threads: " << nb_threads << std::endl;

    cuda_tools::host_shared_ptr<int> tmp(nb_blocks);
    tmp.host_fill(0);

    single_pass_chained_scan<int><<<nb_blocks, nb_threads>>>(buffer.data_, tmp.data_, buffer.size_, tmp.size_);
    cudaDeviceSynchronize();
    kernel_check_error();

    cudaProfilerStop();
}