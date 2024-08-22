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

/* legion programming system library for UPMEM */
#include <realm/upmem/legion_c_upmem.h> 
/* common header between device and host */
#include <common.h> 

typedef struct __DPU_LAUNCH_ARGS {
    char paddd[256];
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

    printf("DEVICE:::: Running daxpy computation with xptr %p, y_ptr %p, z_ptr %p...\n", 
        args->acc_x.ptr(args->rect.lo), args->acc_y.ptr(args->rect.lo), args->acc_z.ptr(args->rect.lo));

  #ifdef INT32
    printf(" alpha = %d \n", args->alpha); 
  #elif DOUBLE
    printf(" alpha = %f \n", args->alpha); 
  #endif

    for (Legion::PointInRectIterator<1> pir(args->rect); pir(); pir++) {
       args->acc_z.write(*pir, args->alpha * args->acc_x[*pir] + args->acc_y[*pir]); 
      //  printf("read %f,\t",args->alpha * args->acc_x[*pir] + args->acc_y[*pir]);
      //  printf("write %f\n",args->acc_z[*pir]);
    }

    return 0;
}