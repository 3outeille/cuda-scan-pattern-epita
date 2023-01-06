#include "to_bench.cuh"

#include "cuda_tools/cuda_error_checking.cuh"
#include "cuda_tools/host_shared_ptr.cuh"

#include <cuda_profiler_api.h>
#include <iostream>
#include <limits>

template <typename T>
static __global__ void single_pass_chained_scan(T* input, T* tmp, T* status_flag, T* status_data, int* block_id, int input_size) {
    int tid = threadIdx.x;
    __shared__ int bid;
    
    // id of each blocks are now based on shcedulation of block and not by blockIdx.x
    // This is to avoid deadlock when block 0 cannot be scheduled on GPU because GPU is full
    if (tid == 0)
        bid = atomicAdd(block_id, 1);
    
    __syncthreads();

    int id = threadIdx.x + (bid * blockDim.x);

    __shared__ int shared_memory[1024];

    if (id < input_size)
        shared_memory[tid] = input[id];

    __syncthreads();

    // Reduce
    for (int stride = 1; stride < blockDim.x; stride *= 2) {
        if (tid % (2 * stride) == 0)
            shared_memory[tid] += shared_memory[tid + stride];
        __syncthreads();
    }

    if (tid == 0)
        tmp[bid] = shared_memory[0];

    __syncthreads();

    // Adjacent synchronization
    if (tid == 0) {

        if (bid == 0) {
            atomicAdd(&status_data[bid], tmp[0]);
            atomicAdd(&status_flag[bid], 1);
        } else {

            int prev_status_flag;

            do {
                prev_status_flag = atomicAdd(&status_flag[bid - 1], 0);
            } while (prev_status_flag == 0);

            int prev_status_data = atomicAdd(&status_data[bid - 1], 0);

            atomicAdd(&status_data[bid], prev_status_data + tmp[bid]);
            atomicAdd(&status_flag[bid], 1);
        }
    }
    
    if (tid == 0 && bid > 0)
        input[bid * blockDim.x] += status_data[bid - 1];

    __syncthreads();

    if (id < input_size)
        shared_memory[tid] = input[id];

    __syncthreads();

    // Scan
    int x;

    for (int stride = 1; stride < blockDim.x; stride *= 2) {
        if (tid >= stride && tid != 0)
            x = shared_memory[tid] + shared_memory[tid - stride];

        __syncthreads();

        if (tid >= stride && tid != 0)
            shared_memory[tid] = x;

        __syncthreads();
    }

    if (id < input_size)
        input[id] = shared_memory[tid];
}

void scan_opti_4(cuda_tools::host_shared_ptr<int> buffer) {
    // Single-pass chained-scan using reduce-then-scan
    // TODO: seems to be veryslow
    constexpr int nb_threads = 1024;
    const int nb_blocks = (buffer.size_ + nb_threads - 1) / nb_threads;

    int* tmp;
    int* status_flag;
    int* status_data;
    int* block_id;

    cudaMalloc(&tmp, nb_blocks * sizeof(int));
    cudaMalloc(&status_flag, nb_blocks * sizeof(int));
    cudaMalloc(&status_data, nb_blocks * sizeof(int));
    cudaMalloc(&block_id, sizeof(int));

    cudaMemset(tmp, 0, nb_blocks * sizeof(int));
    cudaMemset(status_flag, 0, nb_blocks * sizeof(int));
    cudaMemset(status_data, 0, nb_blocks * sizeof(int));
    cudaMemset(block_id, 0, sizeof(int));
    
    cudaProfilerStart();

    single_pass_chained_scan<int><<<nb_blocks, nb_threads>>>(buffer.data_, tmp, status_flag, status_data, block_id, buffer.size_);
    cudaDeviceSynchronize();
    kernel_check_error();

    cudaProfilerStop();
}