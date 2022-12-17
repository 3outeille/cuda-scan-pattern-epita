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

    __syncthreads();

}

template <typename T>
static __global__ void propagate_block(T *buffer, T* tmp)
{
    int id = threadIdx.x + (blockIdx.x * blockDim.x);
    if (blockIdx.x > 0) {
        
        for (int i = 0; i < blockIdx.x; i++)
            buffer[id] += tmp[i];
    }
}

void scan_opti_2(cuda_tools::host_shared_ptr<int> buffer)
{
    // (page 17/21) https://developer.download.nvidia.com/compute/cuda/1.1-Beta/x86_website/projects/scan/doc/scan.pdf
    cudaProfilerStart();

    constexpr int nb_threads = 1024;
    const int nb_blocks = (buffer.size_ + nb_threads - 1) / nb_threads;

    // std::cout << "nb_threads: " << nb_threads << std::endl;
    // std::cout << "nb_blocks: " << nb_blocks << std::endl;

    cuda_tools::host_shared_ptr<int> tmp(nb_blocks);    

    // Compute Scan for each block + store last elt of each scanned block in `tmp`
    scan_block<int><<<nb_blocks, nb_threads>>>(buffer.data_, tmp.data_, buffer.size_);
    cudaDeviceSynchronize();
    // Add tmp[i] to all values of each scanned block [`i+1`, nb_blocks]
    propagate_block<int><<<nb_blocks, nb_threads>>>(buffer.data_, tmp.data_);
    cudaDeviceSynchronize();
    kernel_check_error();

    cudaProfilerStop();
}