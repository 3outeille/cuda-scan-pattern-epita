#pragma once

#include "cuda_tools/host_shared_ptr.cuh"

void baseline_scan(cuda_tools::host_shared_ptr<int> buffer);

void scan_opti_1(cuda_tools::host_shared_ptr<int> buffer);

void scan_opti_2(cuda_tools::host_shared_ptr<int> buffer);

void scan_opti_3(cuda_tools::host_shared_ptr<int> buffer);

void scan_opti_4(cuda_tools::host_shared_ptr<int> buffer);

void scan_opti_4_slow(cuda_tools::host_shared_ptr<int> buffer);

void scan_opti_4_reminder(cuda_tools::host_shared_ptr<int> buffer);

void scan_opti_5(cuda_tools::host_shared_ptr<int> buffer);

void scan_opti_6(cuda_tools::host_shared_ptr<int> buffer);

void scan_opti_7(cuda_tools::host_shared_ptr<int> buffer);