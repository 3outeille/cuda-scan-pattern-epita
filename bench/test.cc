
#include <benchmark/benchmark.h>
#include "test_helpers.hh"
#include "to_bench.cuh"
#include <iostream>

int main(void)
{
    // int n = 64;
    int n = 16777216;
    std::vector<int> vec(n);
    for (int i = 0; i < n; i++)
        vec[i] = 1;
 
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
    scan_opti_1(buffer);
    // Retrieve your result
    int *res = buffer.download();

    // Assert activated only in debug mode
    for (int i = 0; i < n; i++) {
        if (res_baseline[i] != res[i])
        {
            std::cout << "(" << i << ") " << res_baseline[i] << " = " << res[i] << std::endl;
            assert(false);
        }
    }

    std::cout << "OK" << std::endl;

    return 0;
}

// if (tid == 0 && blockIdx.x == 0)
// {
//     printf("(reduce_size: %d) ", reduce_arr_size);
//     for (int i = 0; i < reduce_arr_size; i++)
//         printf("%d ", reduce_arr[i]);
//     printf("\n");
// }

// if (tid == 0 && blockIdx.x == 0)
// {
//     printf("(buffer_size: %d) ", buffer_size);
//     for (int i = 0; i < buffer_size; i++)
//         printf("%d ", buffer[i]);
//     printf("\n");
// }