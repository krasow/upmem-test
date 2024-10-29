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

#define BLOCK_SIZE 1

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
  printf("DEVICE::: my tasklet id is %d, the lower bound of the rect is %d and %d\n", tasklet_id, args->rect.lo.values[0], args->rect.lo.values[1]);
  printf("the low value of rect_x is %d and %d and the high value of rect_x is %d and %d\n", args->rect_x.lo.values[0], args->rect_x.lo.values[1], args->rect_x.hi.values[0], args->rect_x.hi.values[1]);
  printf("the low value of rect_y is %d and %d and the high value of rect_y is %d and %d\n", args->rect_y.lo.values[0], args->rect_y.lo.values[1], args->rect_y.hi.values[0], args->rect_y.hi.values[1]);
  printf("the number of subregions is %d\n", args->num_subregions);

#endif

  // Rect<2> rect;
  // rect.lo.values[0] = args->rect.lo.values[0];
  // rect.lo.values[1] = args->rect.lo.values[1];

  // rect.hi.values[0] = args->rect.hi.values[0];
  // rect.hi.values[1] = args->rect.hi.values[1];

  // AccessorRO block_acc_y;
  // AccessorRO block_acc_x;
  // AccessorWD block_acc_z;

  // set base pointer for the new block accessors
  // printf("%d\n", BLOCK_SIZE * WIDTH * sizeof(TYPE));
  // block_acc_x.accessor.base = (uintptr_t)mem_alloc(BLOCK_SIZE * WIDTH * sizeof(TYPE));
  // block_acc_y.accessor.base = (uintptr_t)mem_alloc(BLOCK_SIZE * WIDTH * sizeof(TYPE));
  // block_acc_z.accessor.base = (uintptr_t)mem_alloc(BLOCK_SIZE * WIDTH * sizeof(TYPE));
  // // set strides from base accessor
  // block_acc_x.accessor.strides = args->acc_x.accessor.strides;
  // block_acc_y.accessor.strides = args->acc_y.accessor.strides;
  // block_acc_z.accessor.strides = args->acc_z.accessor.strides;

  // // iterator through all elements

  // int counter = 0;
  // unsigned int index = tasklet_id;

  // unsigned int total_ele = WIDTH * HEIGHT;

  // for(; index<total_ele; index +=NR_TASKLETS){

  //   unsigned int row = index/WIDTH;
  //   unsigned int col = index%WIDTH;
  //   Legion::PointInRectIterator<2> pir_a(rect);
  //   Legion::PointInRectIterator<2> pir_b(rect);
  //   Legion::PointInRectIterator<2> pir_z(rect);

  //   pir_a += row*WIDTH;
  //   READ_BLOCK(*pir_a, args->acc_x, block_acc_x, WIDTH * BLOCK_SIZE* sizeof(TYPE));

  //   pir_b += col*WIDTH;
  //   READ_BLOCK(*pir_b, args->acc_y, block_acc_y, WIDTH * BLOCK_SIZE* sizeof(TYPE));

  //   Rect<2> block_rect;
  //   block_rect.lo.values[0] = 0;
  //   block_rect.lo.values[1] = 0;
  //   block_rect.hi.values[0] = WIDTH-1;
  //   block_rect.hi.values[1] = WIDTH-1;

  //   TYPE sum=0;

  //   for (Legion::PointInRectIterator<2> pir_block(block_rect); pir_block();
  //        pir_block++) {
  //     printf("(%f, %f) ", block_acc_x[*pir_block] ,block_acc_y[*pir_block]);

  //     sum += block_acc_x[*pir_block] * block_acc_y[*pir_block];
  //     // block_acc_z.write(*pir_block, args->alpha * block_acc_x[*pir_block] +
  //     //                                   block_acc_y[*pir_block]);
  //   }

  //   printf("sum of tasklet \n");
  //   block_rect.lo.values[0] = 0;
  //   block_rect.lo.values[1] = 0;
  //   block_rect.hi.values[0] = 0;
  //   block_rect.hi.values[1] = 0;
  //   Legion::PointInRectIterator<2> pir_block(block_rect);
  //   block_acc_z.write(*pir_block, me());

  //   pir_z += index;

  //   WRITE_BLOCK(*pir_z, args->acc_z, block_acc_z, sizeof(TYPE));
  // }

  // for (Legion::PointInRectIterator<2> pir(rect); pir();
  //      pir += (NR_TASKLETS * WIDTH * BLOCK_SIZE)) {

  //   // read blocks to respective base pointers
  //   // #define READ_BLOCK(point, acc_full, acc_block, bytes)  

  //   READ_BLOCK(*pir, args->acc_x, block_acc_x, WIDTH * BLOCK_SIZE* sizeof(TYPE));
  //   READ_BLOCK(*pir, args->acc_y, block_acc_y, WIDTH * BLOCK_SIZE* sizeof(TYPE));
  //   READ_BLOCK(*pir, args->acc_z, block_acc_z, WIDTH * BLOCK_SIZE* sizeof(TYPE));

  //   Rect<2> block_rect;
  //   block_rect.lo.values[0] = 0;
  //   block_rect.lo.values[1] = 0;
  //   block_rect.hi.values[0] = WIDTH-1;
  //   block_rect.hi.values[1] = WIDTH-1;

  //   TYPE sum=0;
  //   // block iterator
  //   for (Legion::PointInRectIterator<2> pir_block(block_rect); pir_block();
  //        pir_block++) {
  //     // printf("(%f, %f) ", block_acc_x[*pir] ,block_acc_y[*pir]);

  //     sum += block_acc_x[*pir_block] * block_acc_y[*pir_block];
  //     block_acc_z.write(*pir_block, args->alpha * block_acc_x[*pir_block] +
  //                                       block_acc_y[*pir_block]);
  //   }
  //   Legion::PointInRectIterator<2> pir_block(block_rect);
  //   block_acc_z.write(*pir_block, sum);
  //   // write block
  //   // #define WRITE_BLOCK(point, acc_full, acc_block, bytes)
  //   WRITE_BLOCK(*pir, args->acc_z, block_acc_z, WIDTH * BLOCK_SIZE * sizeof(TYPE));

  //   counter++;
  // }
  return 0;
}
