#include "to_bench.cuh"

#include "cuda_tools/cuda_error_checking.cuh"
#include "cuda_tools/host_shared_ptr.cuh"

#include <cuda_profiler_api.h>
#include <iostream>

template <typename T>
static __global__ void scan_block(T *buffer, T* last_elt_of_each_scan_block, int size)
{
    int tid = threadIdx.x;
    int id = threadIdx.x + (blockIdx.x * blockDim.x);

    __shared__ int shared_memory[1024];

    if (id < size)
        shared_memory[tid] = buffer[id];

    __syncthreads();

    int x;

    for (unsigned int stride = 1; stride < size; stride *= 2)
    {
        if (tid >= stride)
            x = shared_memory[tid] + shared_memory[tid - stride];

        __syncthreads();

        if (tid >= stride)
            shared_memory[tid] = x;

        __syncthreads();
    }

    if (tid == blockDim.x - 1)
        last_elt_of_each_scan_block[blockIdx.x] = shared_memory[tid];

    if (id < size)
        buffer[id] = shared_memory[tid];
}

template <typename T>
static __global__ void propagate_block(T *buffer, T* tmp)
{
    int id = threadIdx.x + (blockIdx.x * blockDim.x);
    if (blockIdx.x > 0)
        buffer[id] += tmp[blockIdx.x - 1];
}

void scan_opti_1(cuda_tools::host_shared_ptr<int> buffer)
{
    // (page 17/21) https://developer.download.nvidia.com/compute/cuda/1.1-Beta/x86_website/projects/scan/doc/scan.pdf
    // Scan-then-propagate
    cudaProfilerStart();

    constexpr int nb_threads = 1024;
    const int nb_blocks = (buffer.size_ + nb_threads - 1) / nb_threads;

    int *tmp;
    cudaMalloc(&tmp, nb_blocks * sizeof(int));
    cudaMemset(tmp, 0, nb_blocks * sizeof(int));

    int *tmp2;
    cudaMalloc(&tmp2, nb_blocks * sizeof(int));
    cudaMemset(tmp2, 0, nb_blocks * sizeof(int));

    // Compute Scan for each block + store last elt of each scanned block in `tmp`
    scan_block<int><<<nb_blocks, nb_threads>>>(buffer.data_, tmp, buffer.size_);
    cudaDeviceSynchronize();
    // Compute scan on `tmp` in `tmp2
    const int nb_blocks_2 = (nb_blocks + nb_threads - 1) / nb_threads;
    scan_block<int><<<nb_blocks_2, nb_threads>>>(tmp, tmp2, nb_blocks);
    cudaDeviceSynchronize();

    scan_block<int><<<1, nb_blocks_2>>>(tmp2, tmp2, nb_blocks_2);
    cudaDeviceSynchronize();

    // Add tmp2[i] to all values of scanned block `i+1`
    propagate_block<int><<<nb_blocks_2, nb_threads>>>(tmp, tmp2);
    cudaDeviceSynchronize();

    // Add tmp[i] to all values of scanned block `i+1`
    propagate_block<int><<<nb_blocks, nb_threads>>>(buffer.data_, tmp);
    cudaDeviceSynchronize();
    kernel_check_error();

    cudaProfilerStop();
}