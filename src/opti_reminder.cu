#include "to_bench.cuh"

#include "cuda_tools/cuda_error_checking.cuh"
#include "cuda_tools/host_shared_ptr.cuh"

#include <cuda_profiler_api.h>
#include <iostream>
#include <limits>

template <typename T>
static __global__ void single_pass_chained_scan_reminder(T* input, T* tmp, T* status_flag, T* status_data,
                                                         int input_size, int tmp_size, int status_size) {
    int tid = threadIdx.x;
    int id = threadIdx.x + (blockIdx.x * blockDim.x);

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
        tmp[blockIdx.x] = shared_memory[0];

    __syncthreads();

    // Adjacent synchronization
    if (tid == 0) {

        if (blockIdx.x == 0) {
            atomicAdd(&status_data[blockIdx.x], tmp[0]);
            atomicAdd(&status_flag[blockIdx.x], 1);
        } else {

            int prev_status_data;
            int prev_status_flag;

            // TODO: possible deadlock, cf opti4.cu with `bid`
            do {
                // We must read data before setting flag because:
                prev_status_data = atomicAdd(&status_data[blockIdx.x - 1], 0);
                // Another block may be doing the 2 atomic write at line 54 && 55
                // Resulting in old data being read
                prev_status_flag = atomicAdd(&status_flag[blockIdx.x - 1], 0);
            } while (prev_status_flag == 0);

            atomicAdd(&status_data[blockIdx.x], prev_status_data + tmp[blockIdx.x]);
            atomicAdd(&status_flag[blockIdx.x], 1);
        }
    }

    if (tid == 0 && blockIdx.x > 0)
        input[blockIdx.x * blockDim.x] += status_data[blockIdx.x - 1];

    __syncthreads();

    // Scan
    int x;

    for (int stride = 1; stride < blockDim.x; stride *= 2) {
        if (tid >= stride && (id % blockDim.x) != 0)
            x = input[id] + input[id - stride];

        __syncthreads();

        if (tid >= stride && (id % blockDim.x) != 0)
            input[id] = x;

        __syncthreads();
    }
}

void scan_opti_4_reminder(cuda_tools::host_shared_ptr<int> buffer) {
    // Single-pass chained-scan using reduce-then-scan
    // TODO: seems to be veryslow
    constexpr int nb_threads = 1024;
    const int nb_blocks = (buffer.size_ + nb_threads - 1) / nb_threads;

    cuda_tools::host_shared_ptr<int> tmp(nb_blocks);
    tmp.host_fill(0);

    cuda_tools::host_shared_ptr<int> status_flag(nb_blocks);
    status_flag.host_fill(0);

    cuda_tools::host_shared_ptr<int> status_data(nb_blocks);
    status_data.host_fill(0);

    tmp.upload();
    status_flag.upload();
    status_data.upload();

    cudaProfilerStart();

    single_pass_chained_scan_reminder<int><<<nb_blocks, nb_threads>>>(buffer.data_, tmp.data_, status_flag.data_, status_data.data_,
                                                             buffer.size_, tmp.size_, status_flag.size_);
    cudaDeviceSynchronize();
    kernel_check_error();

    cudaProfilerStop();
}