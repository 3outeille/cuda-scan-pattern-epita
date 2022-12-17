
#include <benchmark/benchmark.h>
#include "test_helpers.hh"
#include "to_bench.cuh"
#include <iostream>

int main(void)
{
    // int n = 4 * 4;
    int n = 1024 * 1024;
    std::vector<int> vec(n);
    for (int i = 0; i < n; i++)
        vec[i] = i;
 
    // Buffer baseline
    cuda_tools::host_shared_ptr<int> buffer_baseline(vec.size());
    buffer_baseline.host_fill(vec);
    // buffer_baseline.print_host_values();
    buffer_baseline.upload();
    // Baseline reduce
    baseline_scan(buffer_baseline);
    // Get result
    int *res_baseline = buffer_baseline.download();

    // Buffer your
    cuda_tools::host_shared_ptr<int> buffer(vec.size());
    buffer.host_fill(vec);
    buffer.upload();
    scan_opti_2(buffer);
    // Retrieve your result
    int *res = buffer.download();

    // Assert activated only in debug mode
    for (int i = 0; i < n; i++) {
        // std::cout << "(" << i << ") " << res_baseline[i] << " = " << res[i] << std::endl;
        assert(res_baseline[i] == res[i]);
    }

    return 0;
}