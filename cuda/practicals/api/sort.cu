#include <algorithm>
#include <iostream>

#include <cuda.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/sort.h>

#include "util.hpp"

int main(int argc, char** argv) {
    size_t pow = read_arg(argc, argv, 1, 16);
    size_t n = 1 << pow;
    auto size_in_bytes = n * sizeof(double);

    std::cout << "sort test of length n = " << n
              << " : " << size_in_bytes/(1024.*1024.) << "MB"
              << std::endl;

    // fill a vector with random values
    thrust::host_vector<double>   values_host(n);
    thrust::device_vector<double> values_device(n);

    // start the nvprof profiling
    cudaProfilerStart();

    // copy values to device
    auto start = get_time();
    values_device = values_host;

    // sort values
    thrust::sort(thrust::device, values_device.begin(), values_device.end());

    auto time_taken = get_time() - start;

    std::cout << "time : " << time_taken << "s\n";

    // copy result back to host
    values_host = values_device;

    // check for errors
    bool pass = std::is_sorted(values_host.begin(), values_host.end());

    // stop the profiling session
    cudaProfilerStop();

    std::cout << (pass ? "passed\n" : "failed\n");

    return 0;
}

