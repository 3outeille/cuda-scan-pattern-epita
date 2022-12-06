#include "host_shared_ptr.cuh"

#include <cstdio>
#include <algorithm>

#include <thrust/execution_policy.h>
#include <thrust/uninitialized_fill.h>

#include "cuda_error_checking.cuh"
#include "template_generator.hh"

namespace cuda_tools
{

template_generation(host_shared_ptr);

template <typename T>
void host_shared_ptr<T>::device_allocate(std::size_t size)
{
    cuda_safe_call(cudaMalloc((void**)&data_, sizeof(T) * size));
}

template <typename T>
host_shared_ptr<T>::host_shared_ptr(std::size_t size) : size_(size)
{
    device_allocate(size);
}

template <typename T>
host_shared_ptr<T>::host_shared_ptr(host_shared_ptr<T>&& ptr) : data_(ptr.data_), host_data_(ptr.host_data_), size_(ptr.size_), counter_(ptr.counter_ + 1)
{}

template <typename T>
host_shared_ptr<T>::host_shared_ptr(host_shared_ptr<T>& ptr) : data_(ptr.data_), host_data_(ptr.host_data_), size_(ptr.size_), counter_(ptr.counter_ + 1)
{}

template <typename T>
host_shared_ptr<T>& host_shared_ptr<T>::operator=(host_shared_ptr<T>&& r)
{
    data_ = r.data_;
    size_ = r.size_;
    counter_ = r.counter_ + 1;
    return *this;
}

template <typename T>
host_shared_ptr<T>::~host_shared_ptr()
{
    if (--counter_ == 0)
    {
        cuda_safe_call(cudaFree(data_));
        if (host_data_ != nullptr)
            delete[] host_data_;
    }
}

template <typename T>
T& host_shared_ptr<T>::operator[](std::ptrdiff_t idx) const noexcept
{
    return host_data_[idx];
}

template <typename T>
T& host_shared_ptr<T>::operator[](std::ptrdiff_t idx) noexcept
{
    return host_data_[idx];
}

template <typename T>
T* host_shared_ptr<T>::download()
{
    if (host_data_ == nullptr)
        host_allocate();
    cuda_safe_call(cudaMemcpy(host_data_, data_, sizeof(T) * size_, cudaMemcpyDeviceToHost));
    return host_data_;
}

template <typename T>
void host_shared_ptr<T>::upload()
{
    cuda_safe_call(cudaMemcpy(data_, host_data_, sizeof(T) * size_, cudaMemcpyHostToDevice));
}

template <typename T>
void host_shared_ptr<T>::device_fill(const T val)
{
    thrust::uninitialized_fill(thrust::device, data_, data_ + size_, val);
}

template <typename T>
void host_shared_ptr<T>::host_allocate()
{
    host_allocate(size_);
}

template <typename T>
void host_shared_ptr<T>::host_allocate(std::size_t size)
{
    // Maybe there is a better option than new ?
    host_data_ = new T[size];
    size_ = size;
}

template <typename T>
void host_shared_ptr<T>::host_fill(const T val)
{
    if (host_data_ == nullptr)
        host_allocate();

    std::transform(host_data_,
                   host_data_ + size_,
                   host_data_,
                   [val]([[maybe_unused]]T arg){return val;});
}

} // namespace cuda_tools