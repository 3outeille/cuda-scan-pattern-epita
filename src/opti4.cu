#include "to_bench.cuh"

#include "cuda_tools/cuda_error_checking.cuh"
#include "cuda_tools/host_shared_ptr.cuh"

#include <cuda_profiler_api.h>
#include <iostream>
#include <limits>

template <typename T>
static __global__ void single_pass_chained_scan(T *buffer, T *reduce_arr, int buffer_size, int reduce_arr_size)
{
    int tid = threadIdx.x;
    int id = threadIdx.x + (blockIdx.x * blockDim.x);

    __shared__ int shared_memory[1024];

    if (id < buffer_size)
        shared_memory[tid] = buffer[id];

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
    if (tid == 0 && blockIdx.x > 0) {
        // Wait until all previous block finish to compute its reduce.
        for (int i = blockIdx.x - 1; i >= 0; i--)
        {
            // Force reduce_arr to be in global memory (and not register)
            while (atomicAdd(&reduce_arr[i], 0) == std::numeric_limits<T>::min())
            {
            }
        }

        // Add associate reduce_arr to 1st element of each block of buffer.
        for (int i = blockIdx.x - 1; i >= 0; i--)
            buffer[blockIdx.x * blockDim.x] += reduce_arr[i];
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
    // Single-pass chained-scan using reduce-then-scan
    // TODO: seems to be veryslow
    cudaProfilerStart();

    constexpr int nb_threads = 1024;
    const int nb_blocks = (buffer.size_ + nb_threads - 1) / nb_threads;

    cuda_tools::host_shared_ptr<int> tmp(nb_blocks);
    tmp.host_fill(std::numeric_limits<int>::min());
    tmp.upload();

    single_pass_chained_scan<int><<<nb_blocks, nb_threads>>>(buffer.data_, tmp.data_, buffer.size_, tmp.size_);
    cudaDeviceSynchronize();
    kernel_check_error();

    cudaProfilerStop();
}