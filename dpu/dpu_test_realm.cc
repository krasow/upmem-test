/* Copyright 2024 Stanford University, Los Alamos National Laboratory,
 *                Northwestern University
 * 
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */


extern "C" {
#include <stdint.h>
#include <defs.h>
#include <mram.h>
#include <alloc.h>
#include <barrier.h>
}

/* realm system library for UPMEM */
#include <realm/upmem/realm_c_upmem.h> 
/* common header between device and host */
#include <common.h> 

#define BLOCK_SIZE 8

typedef struct __DPU_LAUNCH_ARGS {
    char paddd[128];
} __attribute__((aligned(8))) __DPU_LAUNCH_ARGS;

__host __DPU_LAUNCH_ARGS ARGS;  

DPU_LAUNCH_ARGS* args = (DPU_LAUNCH_ARGS*)(&ARGS);

int main_kernel1();

// Barrier
BARRIER_INIT(my_barrier, NR_TASKLETS);

int (*kernels[nr_kernels])(void) = {main_kernel1};

int main(void) {
    return kernels[args->kernel](); 
}

int main_kernel1() {
    unsigned int tasklet_id = me();
    if (tasklet_id == 0) {
        printf("running with %d Tasklets\n", NR_TASKLETS);
    }

    unsigned int row, column;
    row = 64;
    column = 64;

    unsigned total_size = row * column;

    for(unsigned int index = tasklet_id; index < total_size; index += NR_TASKLETS){

        unsigned int idx = index/column;
        unsigned int idy = index%column;

        TYPE sum = 0;

        for(unsigned int k = 0; k < column; k++){
            // .read with true "single" arg will READ for mram a element at a time
            TYPE a = args->arrayA_accessor.read(Point<2>(idx, k), true);
            TYPE b = args->arrayB_accessor.read(Point<2>(k, idy), true);
            sum += a*b;
        }
        // .write with true "single" arg will WRITE for mram a element at a time
        args->arrayC_accessor.write(Point<2>(idx, idy), sum, true);
    }  

    return 0;
}