#include "to_bench.cuh"

#include "cuda_tools/cuda_error_checking.cuh"
#include "cuda_tools/host_shared_ptr.cuh"

#include <cuda_profiler_api.h>
#include <iostream>

template <typename T>
static __global__ void reduce_block(T *buffer, T *reduce_arr, int buffer_size)
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
}

template <typename T>
static __global__ void scan_block(T *reduce_arr, int reduce_arr_size)
{
    int tid = threadIdx.x;
    int id = threadIdx.x + (blockIdx.x * blockDim.x);

    __shared__ int shared_memory[1024];

    if (id < reduce_arr_size)
        shared_memory[tid] = reduce_arr[id];
    else
        shared_memory[tid] = 0;

    __syncthreads();

    // Inclusive scan on reduce_arr
    int x;
    for (int stride = 1; stride < reduce_arr_size; stride *= 2)
    {
        if (tid >= stride)
            x = shared_memory[tid] + shared_memory[tid - stride];

        __syncthreads();

        if (tid >= stride)
            shared_memory[tid] = x;

        __syncthreads();
    }

    if (id < reduce_arr_size)
        reduce_arr[id] = shared_memory[tid];
}

template <typename T>
static __global__ void propagate(T *buffer, T *reduce_arr, int buffer_size)
{
    int tid = threadIdx.x;
    int id = threadIdx.x + (blockIdx.x * blockDim.x);

    // Add associate reduce_arr to 1st element of each block of buffer.
    if (blockIdx.x > 0 && tid == 0)
        atomicAdd(&buffer[blockIdx.x * blockDim.x], reduce_arr[blockIdx.x - 1]);

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

void scan_opti_3(cuda_tools::host_shared_ptr<int> buffer)
{
    // Reduce-then-scan (reduce-scan-scan)
    cudaProfilerStart();

    constexpr int nb_threads = 1024;
    const int nb_blocks = (buffer.size_ + nb_threads - 1) / nb_threads;

    int *tmp;
    cudaMalloc(&tmp, nb_blocks * sizeof(int));
    cudaMemset(tmp, 0, nb_blocks * sizeof(int));

    reduce_block<int><<<nb_blocks, nb_threads>>>(buffer.data_, tmp, buffer.size_);
    cudaDeviceSynchronize();
    scan_block<int><<<1, nb_blocks>>>(tmp, nb_blocks);
    cudaDeviceSynchronize();
    propagate<int><<<nb_blocks, nb_threads>>>(buffer.data_, tmp, buffer.size_);
    cudaDeviceSynchronize();
    kernel_check_error();

    cudaProfilerStop();
}