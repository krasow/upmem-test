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
#include <alloc.h>
#include <barrier.h>
#include <defs.h>
#include <mram.h>
#include <stdint.h>
}

/* legion programming system library for UPMEM */
#include <realm/upmem/legion_c_upmem.h>
/* common header between device and host */
#include <common.h>

typedef struct __DPU_LAUNCH_ARGS {
  char paddd[512];
} __attribute__((aligned(8))) __DPU_LAUNCH_ARGS;

__host __DPU_LAUNCH_ARGS ARGS;

DPU_LAUNCH_ARGS *args = (DPU_LAUNCH_ARGS *)(&ARGS);

int main_kernel1();

// Barrier
BARRIER_INIT(my_barrier, NR_TASKLETS);

int (*kernels[nr_kernels])(void) = {main_kernel1};

int main(void) { return kernels[args->kernel](); }

int main_kernel1() {
  unsigned int tasklet_id = me();

#ifdef PRINT_UPMEM
  printf("DEVICE::: my tasklet id is %d, the lower bound of the rect is %lu \n", tasklet_id, args->rect.lo.value);

  // if (tasklet_id == 0) {
  //   printf("DEVICE:::: Running mat multiplication with xptr %p, y_ptr %p, z_ptr "
  //          "%p...",
  //          args->acc_x.ptr(args->rect.lo), args->acc_y.ptr(args->rect.lo),
  //          args->acc_z.ptr(args->rect.lo));
  // }


#endif
  if(tasklet_id == 0){
  printf("rect.lo %d \n", args->rect.lo.value);
  }
  //print the matrix
  Rect<1> rect;
  rect.lo = args->rect.lo;
  rect.hi = args->rect.hi;

  unsigned int range = rect.hi.value - rect.lo.value;
  unsigned int index = rect.lo.value;

  AccessorRO block_acc_y;
  AccessorRO block_acc_x;
  block_acc_x.accessor.base = (uintptr_t)mem_alloc((1 + WIDTH) * sizeof(TYPE));
  block_acc_y.accessor.base = (uintptr_t)mem_alloc((1 + WIDTH) * sizeof(TYPE));
  // set strides from base accessor
  block_acc_x.accessor.strides = args->acc_x.accessor.strides;
  block_acc_y.accessor.strides = args->acc_y.accessor.strides;

  unsigned int curr_row = 0;
  unsigned int curr_col = 0;

  for(unsigned int counter = tasklet_id; counter < range; counter += NR_TASKLETS*WIDTH) {
    //read data
    Rect<1> temp_rect;
    temp_rect.lo = 0;
    temp_rect.hi = WIDTH*HEIGHT;
    Legion::PointInRectIterator<1> pir_a(temp_rect);
    Legion::PointInRectIterator<1> pir_b(temp_rect);
    // Legion::PointInRectIterator<1> pir_z(rect); 
    curr_row = counter/WIDTH;
    curr_col = counter%WIDTH;

    pir_a += curr_row*WIDTH;
    pir_b += curr_col*WIDTH;
    READ_BLOCK(*pir_a, args->acc_x, block_acc_x, WIDTH * sizeof(TYPE));
    READ_BLOCK(*pir_b, args->acc_y, block_acc_y, WIDTH * sizeof(TYPE));

    //calculation
    Rect<1> block_rect;
    block_rect.lo = 0;
    block_rect.hi = WIDTH-1;
    TYPE sum=0;
    // printf("the subregion size")
    for (Legion::PointInRectIterator<1> pir_block(block_rect); pir_block();
         pir_block++) {
      printf("(%f, %f) ", block_acc_x[*pir_block] ,block_acc_y[*pir_block]);
      sum += block_acc_x[*pir_block] * block_acc_y[*pir_block];
    }

    printf("\n");
    
    args->acc_z.accessor.write(Point<1>(counter), sum, true);

  }
  return 0;
}