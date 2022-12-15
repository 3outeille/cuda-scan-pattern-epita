#include "to_bench.cuh"

#include "cuda_tools/cuda_error_checking.cuh"
#include "cuda_tools/host_shared_ptr.cuh"

#include <cuda_profiler_api.h>

template <typename T>
__global__
void kernel_scan_opti_1(T* buffer, int size)
{
    for (int i = 1; i < size; ++i)
        buffer[i] += buffer[i - 1];
}

void scan_opti_1(cuda_tools::host_shared_ptr<int> buffer)
{
    cudaProfilerStart();

	kernel_scan_opti_1<int><<<1, 1>>>(buffer.data_, buffer.size_);

    cudaDeviceSynchronize();
    kernel_check_error();
    
    cudaProfilerStop();
}