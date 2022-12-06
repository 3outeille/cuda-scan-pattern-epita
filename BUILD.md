# Requirements

* [Cuda Toolkit](https://developer.nvidia.com/cuda-downloads)
* C++ compiler ([g++](https://gcc.gnu.org/) for linux,  [MSVC](https://visualstudio.microsoft.com/downloads/) for Windows)
* [GPU supported by CUDA](https://en.wikipedia.org/wiki/CUDA#GPUs_supported)
* [CMake](https://cmake.org/download/)

### Additional libraries

These libraries are included by fetch content. Do not install them yourself.
**Fetch content will do the job for you.**

* [GoogleBenchmark](https://github.com/google/benchmark)

## Build

- To build, execute the following commands :

```bash
mkdir build && cd build
cmake ..
make -j
```

## Run (from ./build directory) :

### Running the benchmarks

```bash
./bench
```

### Running Nsight Compute

- The following command will generate the Nsight Compute report with all kernel information (full).

```bash
ncu -o scan_nsight -f --set full ./bin/bench --bench-nsight
```

You can now open the *.ncu-rep file using Nsight Compute and analyze the results.

- On the OpenStack server, to run Nsight-Compute you should use the following :

To run Nsight Compute to generate a report :
```bash
$(NIXPKGS_ALLOW_UNFREE=1 nix eval --impure --raw nixpkgs#cudaPackages.nsight_compute)/nsight-compute/2022.1.1/ncu
```

To run the UI of Nsight Compute :
```bash
$(NIXPKGS_ALLOW_UNFREE=1 nix eval --impure --raw nixpkgs#cudaPackages.nsight_compute)/nsight-compute/2022.1.1/ncu-ui
```

### Additional infos

* By default the program **will run in release** when it's inside a `build` or `build_release` folder. To build in **debug**, build the project inside a `build_debug` folder.

* You can specify the "--no-check" option when running the bench binary to disable result checking :
```bash
./bin/bench --no-check
```

* You can specify the "--bench-nsight" option when running the bench binary to forbid Google Benchmark from running the functions multiple times (Nsight will do this job) :
```bash
./bin/bench --bench-nsight
```