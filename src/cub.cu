#include "to_bench.cuh"

#include "cuda_tools/cuda_error_checking.cuh"
#include "cuda_tools/host_shared_ptr.cuh"

#include <cub/cub.cuh>
#include <cuda_profiler_api.h>
#include <iostream>
#include <limits>

void scan_cub(cuda_tools::host_shared_ptr<int> buffer) {

    // Determine temporary device storage requirements for inclusive prefix sum
    void* d_temp_storage = NULL;
    size_t temp_storage_bytes = 0;
    
    // When d_temp_storage is NULL, no work is done and the required allocation size is returned in temp_storage_bytes.
    cub::DeviceScan::InclusiveSum(d_temp_storage, temp_storage_bytes, buffer.data_, buffer.data_, buffer.size_);
    // Allocate temporary storage for inclusive prefix sum
    cuda_safe_call(cudaMalloc(&d_temp_storage, temp_storage_bytes));
    
    cudaProfilerStart();
    // Run inclusive prefix sum
    cub::DeviceScan::InclusiveSum(d_temp_storage, temp_storage_bytes, buffer.data_, buffer.data_, buffer.size_);
    cudaProfilerStop();

    cuda_safe_call(cudaFree(d_temp_storage));
}
