//
// Created by vitowu on 2/24/20.
//

#include <iostream>
#include <stdlib.h>
#include <time.h>

#include "kernel.h"
#include "cuda_utils.h"

// 2 ^ 19 iteration time
#define MESSAGE_NUM 0x800000

int main() {
    // randomly generate MESSAGE_NUM states
    auto *init_states = (uint32_t *) malloc(25 * MESSAGE_NUM * sizeof(uint32_t));
    srand(time(0));
    for (uint32_t i = 0; i < 25 * MESSAGE_NUM; i++)
        init_states[i] = (uint32_t) rand();

    // launch gpu kernel
    launch_keccak_kernel(0, init_states, MESSAGE_NUM);

    // release temporary buffer
    free(init_states);

    return 0;
}