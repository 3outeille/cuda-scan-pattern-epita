#include "test_helpers.hh"

#include "cuda_tools/host_shared_ptr.cuh"

#include <algorithm>
#include <benchmark/benchmark.h>
#include <gtest/gtest.h>

template <typename FUNC>
void check_buffer(cuda_tools::host_shared_ptr<int> buffer,
                  FUNC func,
                  benchmark::State& st)
{
    int* host_buffer = buffer.download();

    if (!std::all_of(host_buffer,
                     host_buffer + buffer.size_,
                     func))
    {
        std::cout << "Expected " << func(0) << ", got " << *host_buffer << std::endl;
        st.SkipWithError("Failed test");
    }
}

void check_buffer(cuda_tools::host_shared_ptr<int> buffer,
                  cuda_tools::host_shared_ptr<int> expected,
                  benchmark::State& st)
{
    int* host_buffer = buffer.download();

    if (!std::equal(host_buffer,
                    host_buffer + buffer.size_,
                    expected.host_data_))
    {
        auto [first, second] = std::mismatch(host_buffer,
                                             host_buffer + buffer.size_,
                                             expected.host_data_);
        std::cout << "Error at " << first - host_buffer << ": " << *first << " "
                  << *second << std::endl;
        st.SkipWithError("Failed test");
    }
}