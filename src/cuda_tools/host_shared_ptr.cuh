#pragma once

#include <cstddef>
#include <functional>

namespace cuda_tools
{
template <typename T>
struct host_shared_ptr
{
    host_shared_ptr() = default;
    host_shared_ptr(std::size_t size);
    host_shared_ptr(host_shared_ptr<T>&& ptr);
    host_shared_ptr(host_shared_ptr<T>& ptr);
    host_shared_ptr& operator=(host_shared_ptr<T>&& r);
    
    ~host_shared_ptr();

    T& operator[](std::ptrdiff_t idx) const noexcept;
    T& operator[](std::ptrdiff_t idx) noexcept;
    
    void host_allocate(std::size_t size);
    void host_allocate();
    void host_fill(const T val);
    void device_allocate(std::size_t size);
    void device_fill(const T val);
    T* download();
    void upload();

  
    T* __restrict__ data_ = nullptr;
    T* __restrict__ host_data_ = nullptr;
    std::size_t size_ = 0; 
    int counter_ = 1;
};

} // namespace cuda_tools