#pragma once

#include <chrono>
#include <cuda_profiler_api.h>

#define UNREACHABLE() { \
    printf("file %s line %i: unreachable!\n", __FILE__, __LINE__); \
    fflush(stdout); \
    exit(1); \
}

#define checkCudaErrors(stmt) {                                 \
    cudaError_t err = stmt;                            \
    if (err != cudaSuccess) {                          \
    fprintf(stderr, "%s in file %s, function %s, line %i: %04d %s\n", #stmt, __FILE__, __FUNCTION__, __LINE__, err, cudaGetErrorString(err)); \
    exit(1); \
    }                                                  \
}
