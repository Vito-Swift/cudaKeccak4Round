//
// Created by vitowu on 2/24/20.
//

#ifndef CUDAKECCAK_KERNEL_H
#define CUDAKECCAK_KERNEL_H

#include <stdint.h>

void launch_keccak_kernel(uint32_t dev_id, uint32_t* states, uint32_t state_num);

#endif //CUDAKECCAK_KERNEL_H
