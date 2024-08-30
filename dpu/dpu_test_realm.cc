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

    Rect<1> rect;
    rect.lo = args->rect.lo + tasklet_id;
    rect.hi = args->rect.hi;

    Rect<1> rect_z;
    rect_z.lo = 0;
    rect_z.hi = args->width;

    size_t cnt = 0;
    for(Legion::PointInRectIterator<1> pir_z(rect); pir_z(); pir_z += NR_TASKLETS){
      rect.lo = args->rect.lo + cnt;
      rect.hi = args->rect.hi;
      auto sum = args->alpha;
      sum -= args->alpha;
      Rect<1> rect_y;
      rect_y.lo = 0;
      rect_y.hi = args->height;
      Legion::PointInRectIterator<1> pir_y(rect_y);
      for (Legion::PointInRectIterator<1> pir(rect); pir(); pir+= args->height) {
        sum += args->acc_x[*pir] * args->acc_y[*pir_y];
        // args->acc_z.write(*pir, args->alpha * args->acc_x[*pir] + args->acc_y[*pir]); 
        //  printf("read %f,\t",args->alpha * args->acc_x[*pir] + args->acc_y[*pir]);
        //  printf("write %f\n",args->acc_z[*pir]);
        pir_y++;
      }
      args->acc_z.write(*pir_z, sum);
      cnt+=NR_TASKLETS;
    }

    // for (Legion::PointInRectIterator<1> pir(rect); pir(); pir += NR_TASKLETS) {
    //   args->acc_z.write(*pir, args->alpha * args->acc_x[*pir] + args->acc_y[*pir]); 
    //   //  printf("read %f,\t",args->alpha * args->acc_x[*pir] + args->acc_y[*pir]);
    //   //  printf("write %f\n",args->acc_z[*pir]);
    // }

    return 0;
}