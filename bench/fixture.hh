#pragma once

#include <benchmark/benchmark.h>
#include <numeric>

class Fixture
{
  public:
    static bool no_check;

    template <typename FUNC, typename... Args>
    void bench_scan(benchmark::State& st,
                      FUNC callback,
                      int size,
                      Args&&... args)
    {
        constexpr int val = 1;
        cuda_tools::host_shared_ptr<int> buffer(size);

        // buffer.device_fill(val);

        for (auto _ : st)
        {
            st.PauseTiming();
            buffer.device_fill(val);
            st.ResumeTiming();
            callback(buffer);
        }

        st.SetBytesProcessed(int64_t(st.iterations()) *
                             int64_t(size * sizeof(int)));

        cuda_tools::host_shared_ptr<int> expected(size);
        expected.host_allocate();
        std::iota(expected.host_data_, expected.host_data_ + size, 1);
        if (!no_check)
            check_buffer(buffer, expected, st);
    }

    template <typename FUNC>
    void register_scan(benchmark::State &st, FUNC func)
    {
        int size = st.range(0);
        this->bench_scan(st, func, size);
    }
};

bool Fixture::no_check = false;