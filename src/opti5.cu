#include "to_bench.cuh"

#include "cuda_tools/cuda_error_checking.cuh"
#include "cuda_tools/host_shared_ptr.cuh"

#include <cuda_profiler_api.h>
#include <iostream>
#include <limits>

template <typename T, int BLOCK_SIZE>   
// Single pass chained scan Scan-then-propagate
static __global__ void single_pass_chained_scan_stp(T* input, T* status_flag, T* status_data, int* block_id, int input_size) {
    int tid = threadIdx.x;
    __shared__ int bid;
    
    // id of each blocks are now based on shcedulation of block and not by blockIdx.x
    // This is to avoid deadlock when block 0 cannot be scheduled on GPU because GPU is full
    if (tid == 0)
        bid = atomicAdd(block_id, 1);
    
    __syncthreads();

    int id = threadIdx.x + (bid * blockDim.x);

    __shared__ int shared_memory[BLOCK_SIZE];

    if (id < input_size)
        shared_memory[tid] = input[id];
    else
        shared_memory[tid] = 0; // To avoid undefined behavior

    __syncthreads();

    // Scan
    #pragma unroll
    for (int step = 0; (1 << step) < BLOCK_SIZE; step += 1)
    {
        int left = 1 << step;
        if ((tid & left) != 0) {
            int right = tid >> step;
            int from = left * right - 1;
            shared_memory[tid] += shared_memory[from];
        }

        __syncthreads();
    }

    __syncthreads();


    // Adjacent synchronization
    __shared__ int value_to_propagate;

    if (tid == 0) {
        int block_sum = shared_memory[BLOCK_SIZE - 1];

        if (bid == 0) {
            atomicAdd(&status_data[bid], block_sum);
            atomicAdd(&status_flag[bid], 1);

            value_to_propagate = 0;
        } else {

            int prev_status_flag;

            do {
                prev_status_flag = atomicAdd(&status_flag[bid - 1], 0);
            } while (prev_status_flag == 0);

            int prev_status_data = atomicAdd(&status_data[bid - 1], 0);

            atomicAdd(&status_data[bid], prev_status_data + block_sum);
            atomicAdd(&status_flag[bid], 1);

            value_to_propagate = prev_status_data;
        }
    }

    __syncthreads();

    // Propagate
    if (id < input_size)
        input[id] = shared_memory[tid] + value_to_propagate;
}

void scan_opti_5(cuda_tools::host_shared_ptr<int> buffer) {
    // Single-pass chained-scan using scan_then_propagate
    constexpr int nb_threads = 1024;
    const int nb_blocks = (buffer.size_ + nb_threads - 1) / nb_threads;

    int* status_flag;
    int* status_data;
    int* block_id;

    cudaMalloc(&status_flag, nb_blocks * sizeof(int));
    cudaMalloc(&status_data, nb_blocks * sizeof(int));
    cudaMalloc(&block_id, sizeof(int));

    cudaMemset(status_flag, 0, nb_blocks * sizeof(int));
    cudaMemset(status_data, 0, nb_blocks * sizeof(int));
    cudaMemset(block_id, 0, sizeof(int));
    
    cudaProfilerStart();

    // Single pass chained scan Scan-then-propagate
    single_pass_chained_scan_stp<int, nb_threads><<<nb_blocks, nb_threads>>>(buffer.data_, status_flag, status_data, block_id, buffer.size_);
    cudaDeviceSynchronize();
    kernel_check_error();

    cudaProfilerStop();
}