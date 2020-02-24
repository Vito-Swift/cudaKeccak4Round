//
// Created by vitowu on 2/24/20.
//

#ifndef CUDAKECCAK_CUDA_UTILS_H
#define CUDAKECCAK_CUDA_UTILS_H

#include <stdio.h>
#include <stdint.h>

#include <cuda_runtime.h>

__host__ void
cuda_check(cudaError_t err, const char *file, const int line, bool fatal);

/* a macro that fills in source file and line number */
#define CUDA_CHECK(call) do { \
    cuda_check((call), __FILE__, __LINE__, 1); \
} while(0)

#define MBFLOAT         (1024.0 * 1024.0)

/* check and print info of a GPU device */
__host__ void
check_gpu(const cudaDeviceProp *const dev_prop);

#endif //CUDAKECCAK_CUDA_UTILS_H
