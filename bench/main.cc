#include "test_helpers.hh"
#include "to_bench.cuh"
#include "benchmark_registerer.hh"
#include "fixture.hh"
#include "cuda_tools/host_shared_ptr.cuh"

#include <benchmark/benchmark.h>
#include <tuple>

template <typename Tuple>
constexpr auto tuple_length(Tuple) { return std::tuple_size_v<Tuple>; }

int main(int argc, char** argv)
{
    // Google bench setup
    using benchmark_t = benchmark::internal::Benchmark;
    ::benchmark::Initialize(&argc, argv);
    bool bench_nsight = false;

    // Argument parsing
    for (int i = 1; i < argc; i++)
    {
        if (argv[i] == std::string_view("--no-check"))
        {
            Fixture::no_check = true;
            std::swap(argv[i], argv[--argc]);
        }
        // Set iteration number to 1 not to mess with nsight
        if (argv[i] == std::string_view("--bench-nsight"))
        {
            bench_nsight = true;
            std::swap(argv[i], argv[--argc]);
        }
    }

    // Benchmarks registration
    Fixture fx;
    {
        // Add the sizes to benchmark here
        // TODO : start with 256 to do a block scan then uncomment the bigger size for the grid scan
        constexpr std::array sizes = {
            256,
            //1048576
        };
        
        // Add the name and function to benchmark here
        // TODO
        constexpr std::tuple scan_to_bench
        {
            "Baseline_scan", &baseline_scan,
            "Your_scan", &your_scan,
        };

        //  / 2 because we store name + function pointer
        benchmark_t *b[tuple_length(scan_to_bench) / 2];
        int function_index = 0;

        // Call to registerer
        registerer_scan(&fx, b, function_index, sizes, bench_nsight, scan_to_bench);
    }
    ::benchmark::RunSpecifiedBenchmarks();
}
