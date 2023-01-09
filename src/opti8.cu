#include "to_bench.cuh"

#include "cuda_tools/cuda_error_checking.cuh"
#include "cuda_tools/host_shared_ptr.cuh"

#include <cuda_profiler_api.h>
#include <iostream>
#include <limits>

template <typename T, int BLOCK_SIZE>
// Single pass chained scan Scan-then-propagate
static __global__ void decoupled_look_back_stp_fast(T* input, T* flag, T* record_step_value, int* block_id, int input_size,
                                                    int nb_step) {
    int tid = threadIdx.x;
    __shared__ int bid;

    // id of each blocks are now based on shcedulation of block and not by blockIdx.x
    // This is to avoid deadlock when block 0 cannot be scheduled on GPU because GPU is full
    if (tid == 0)
        bid = atomicAdd(block_id, 1);

    __syncthreads();

    int gid = threadIdx.x + (bid * blockDim.x);

    __shared__ int shared_memory[BLOCK_SIZE];

    if (gid < input_size)
        shared_memory[tid] = input[gid];
    else
        shared_memory[tid] = 0; // To avoid undefined behavior

    __syncthreads();

// Scan
#pragma unroll
    for (int step = 0; (1 << step) < BLOCK_SIZE; step += 1) {
        int left = 1 << step;
        if ((tid & left) != 0) {
            int right = tid >> step;
            int from = left * right - 1;
            shared_memory[tid] += shared_memory[from];
        }

        __syncthreads();
    }

    // Decoupled look-back (Sklansky scan)
    __shared__ int value_to_propagate;

    if (tid == 0) {
        int current_value = shared_memory[BLOCK_SIZE - 1];

        const int nb_flag = nb_step + 1;

        // Write local sum to step 0
        atomicAdd(&record_step_value[bid * nb_flag], current_value);

        // Set flag to A
        atomicExch(&flag[bid], 1);

        for (int step = 0; step < nb_step; step += 1) {
            int left = 1 << step;
            if ((bid & left) != 0) {
                int right = bid >> step;
                int from = left * right - 1;

                // Wait for previous block to be done
                while (atomicAdd(&flag[from], 0) < step + 1) {
                }

                // Add value to current value
                int from_value = atomicAdd(&record_step_value[from * nb_flag + step], 0);
                current_value += from_value;

                // Write current value to step value
                atomicAdd(&record_step_value[bid * nb_flag + step + 1], current_value);

                atomicExch(&flag[bid], step + 2);
            }
        }

        value_to_propagate = current_value - shared_memory[BLOCK_SIZE - 1];
    }

    __syncthreads();

    if (gid < input_size)
        input[gid] = shared_memory[tid] + value_to_propagate;
}

void scan_opti_8(cuda_tools::host_shared_ptr<int> buffer) {
    // Decoupled look back using scan_then_propagate
    constexpr int nb_threads = 1024;
    // constexpr int nb_threads = 2;
    const int nb_blocks = (buffer.size_ + nb_threads - 1) / nb_threads;
    const int nb_step = std::log2(nb_blocks);
    const int nb_flags = nb_step + 1;

    // 0: X / 1: A / N: step N
    int* flag;
    int* record_step_value;
    int* block_id;

    cudaMalloc(&flag, nb_blocks * sizeof(int));
    cudaMalloc(&record_step_value, nb_blocks * nb_flags * sizeof(int));
    cudaMalloc(&block_id, sizeof(int));

    cudaMemset(flag, 0, nb_blocks * sizeof(int));
    cudaMemset(record_step_value, 0, nb_blocks * nb_flags * sizeof(int));
    cudaMemset(block_id, 0, sizeof(int));

    cudaProfilerStart();

    // Single pass chained scan Scan-then-propagate
    decoupled_look_back_stp_fast<int, nb_threads>
        <<<nb_blocks, nb_threads>>>(buffer.data_, flag, record_step_value, block_id, buffer.size_, nb_step);
    cudaDeviceSynchronize();
    kernel_check_error();

    cudaProfilerStop();
}
