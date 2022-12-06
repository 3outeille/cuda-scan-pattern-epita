#pragma once

#include "cuda_tools/host_shared_ptr.cuh"

void baseline_scan(cuda_tools::host_shared_ptr<int> buffer);

void your_scan(cuda_tools::host_shared_ptr<int> buffer);