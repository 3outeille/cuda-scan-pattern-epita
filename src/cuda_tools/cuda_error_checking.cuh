#pragma once

#include <string>

#ifdef NDEBUG
#define kernel_check_error()
#else
#define kernel_check_error()                                                 \
{                                                                            \
	auto e = cudaGetLastError();                                             \
	if (e != cudaSuccess)                                                    \
	{                                                                        \
		std::string error = "Cuda failure in ";                              \
		error += __FILE__;                                                   \
		error += " at line ";                                                \
		error += std::to_string(__LINE__);                                   \
		error += ": ";                                                       \
		error += cudaGetErrorString(e);                                      \
		exit(-1);                                                            \
	}												                         \
}
#endif

#ifdef NDEBUG
#define cuda_safe_call(ans) ans
#else
#define cuda_safe_call(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess)
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}
#endif