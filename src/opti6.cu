#include "to_bench.cuh"

#include "cuda_tools/cuda_error_checking.cuh"
#include "cuda_tools/host_shared_ptr.cuh"

#include <cuda_profiler_api.h>
#include <iostream>
#include <limits>

template <typename T, int BLOCK_SIZE>
// Single pass chained scan Scan-then-propagate
static __global__ void decoupled_look_back_rts(T* input, T* flag, T* record_local_sum, T* record_total_cum_sum, int* block_id,
                                                    int input_size) {
     int tid = threadIdx.x;
    __shared__ int bid;
    
    // id of each blocks are now based on shcedulation of block and not by blockIdx.x
    // This is to avoid deadlock when block 0 cannot be scheduled on GPU because GPU is full
    if (tid == 0)
        bid = atomicAdd(block_id, 1);
    
    __syncthreads();

    int gid = threadIdx.x + (bid * blockDim.x);

    __shared__ int shared_memory[1024];

    if (gid < input_size)
        shared_memory[tid] = input[gid];

    __syncthreads();

    // Reduce
    for (int stride = 1; stride < blockDim.x; stride *= 2) {
        if (tid % (2 * stride) == 0)
            shared_memory[tid] += shared_memory[tid + stride];
        __syncthreads();
    }

    __syncthreads();
    // Decoupled look-back (adjacent synchronization)
    if (tid == 0) {

        if (bid == 0) {
            // First block to be scheduled on SM must have the total sum (since it cannot look-back)
            atomicAdd(&record_total_cum_sum[0], shared_memory[0]);
            atomicExch(&flag[0], 2); // Set to P
        } else {
            // Block has computed its local reduce
            atomicAdd(&record_local_sum[bid], shared_memory[0]);
            atomicAdd(&record_total_cum_sum[bid], shared_memory[0]);
            atomicExch(&flag[bid], 1); // set to A

            int curr_bid = bid;
            int prev_bid = curr_bid - 1;

            while (true) {

                int prev_flag;
                // Look at previous immediate block
                do {
                    prev_flag = atomicAdd(&flag[prev_bid], 0);
                } while (prev_flag == 0);

                if (prev_flag == 2) {
                    // The block has the total sum, we can stop here
                    int prev_record_total_cum_sum = atomicAdd(&record_total_cum_sum[prev_bid], 0);
                    atomicAdd(&record_total_cum_sum[curr_bid], prev_record_total_cum_sum);
                    atomicExch(&flag[curr_bid], 2); // Set to P
                    break;
                } else {
                    // The block has only its local sum, we need to look-back
                    int prev_record_local_sum = atomicAdd(&record_local_sum[prev_bid], 0);
                    atomicAdd(&record_total_cum_sum[curr_bid], prev_record_local_sum);
                    prev_bid -= 1;
                    continue;
                }
            }
        }
    }

    if (tid == 0 && bid > 0)
        input[bid * blockDim.x] += record_total_cum_sum[bid] - shared_memory[0];

    if (gid < input_size)
        shared_memory[tid] = input[gid];

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

    if (gid < input_size)
        input[gid] = shared_memory[tid];
}

void scan_opti_6(cuda_tools::host_shared_ptr<int> buffer) {
    // decouple look-back with reduce-then-scan
    // TODO: understand why it does not have the same perf as decoupled look back with scan-then-propagate
    constexpr int nb_threads = 1024;
    // constexpr int nb_threads = 10;
    const int nb_blocks = (buffer.size_ + nb_threads - 1) / nb_threads;

    // 0: X / 1: A / 2: P
    int* flag;
    int* record_local_sum;
    int* record_total_cum_sum;
    int* block_id;

    cudaMalloc(&flag, nb_blocks * sizeof(int));
    cudaMalloc(&record_local_sum, nb_blocks * sizeof(int));
    cudaMalloc(&record_total_cum_sum, nb_blocks * sizeof(int));
    cudaMalloc(&block_id, sizeof(int));

    cudaMemset(flag, 0, nb_blocks * sizeof(int));
    cudaMemset(record_local_sum, 0, nb_blocks * sizeof(int));
    cudaMemset(record_total_cum_sum, 0, nb_blocks * sizeof(int));
    cudaMemset(block_id, 0, sizeof(int));

    cudaProfilerStart();

    // Single pass chained scan Scan-then-propagate
    decoupled_look_back_rts<int, nb_threads>
        <<<nb_blocks, nb_threads>>>(buffer.data_, flag, record_local_sum, record_total_cum_sum, block_id, buffer.size_);
    cudaDeviceSynchronize();
    kernel_check_error();

    cudaProfilerStop();
}