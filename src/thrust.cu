#include "to_bench.cuh"

#include "cuda_tools/cuda_error_checking.cuh"
#include "cuda_tools/host_shared_ptr.cuh"

#include <thrust/scan.h>
#include <thrust/device_vector.h>
#include <cuda_profiler_api.h>
#include <iostream>
#include <limits>

void scan_thrust(cuda_tools::host_shared_ptr<int> buffer) {

    thrust::device_vector<int> d_buffer(buffer.data_, buffer.data_ + buffer.size_);
    
    cudaProfilerStart();
    thrust::inclusive_scan(d_buffer.begin(), d_buffer.end(), d_buffer.begin());
    cudaProfilerStop();

    thrust::copy(d_buffer.begin(), d_buffer.end(), buffer.data_);
}
