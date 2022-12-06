# TP scan

The goal of the scan TP is to learn how to program scan on GPU and how to program the Decoupled Look-back.

You should first try to have a working block scan before going into the Decoupled Look-back.
You will find "TODO" where you need to modify things and add your code.

## To add and benchmark your scan

In `bench/main.cc`:
* Add the sizes you want to benchmark to "sizes" array
* Add the name / function you want to benchmark to "scan_to_bench" array

In `src/to_bench.cu(h)`:
* Add your functions to benchmark