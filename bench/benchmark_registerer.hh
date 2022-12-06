#pragma once

#include "fixture.hh"

#include <benchmark/benchmark.h>

using namespace std::placeholders;
using benchmark_t = benchmark::internal::Benchmark;

template <typename N, typename FUNC, typename... Args>
void _registerer_scan(Fixture* fx, benchmark_t *b[], int& index, const auto& sizes, bool bench_nsight, N name, FUNC bench_func, Args&&... args)
{
    // Register benchmark
    b[index] = benchmark::RegisterBenchmark(name, std::bind(&Fixture::register_scan<FUNC>, fx, _1, bench_func));
    if (bench_nsight)
        b[index]->UseRealTime()->Unit(benchmark::kMillisecond)->Iterations(1);
    else
        b[index]->UseRealTime()->Unit(benchmark::kMillisecond);

    for (auto size : sizes)
        b[index]->Args({size});

    // Expand tuple (now variadic) to register next until empty
    if constexpr (sizeof...(args) >= 2)
    {
        _registerer_scan<N, FUNC>(fx, b, ++index, sizes, bench_nsight, std::forward<Args>(args)...);
    }
}

template <typename Tuple>
void registerer_scan(Fixture* fx, benchmark_t *b[], int& index, const auto& sizes, bool bench_nsight, Tuple tuple)
{
    // Expand tuple into variadic template
    std::apply([fx, &b, &index, &sizes, bench_nsight](auto &&... args) { _registerer_scan(fx, b, index, sizes, bench_nsight, std::forward<decltype(args)>(args)...); }, tuple);
}