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
// #define NUM_SUBREGIONS 2

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
  if (tasklet_id == 0) {
    printf("DEVICE:::: Running mat multiplication with xptr %p, y_ptr %p, z_ptr "
           "%p...",
           args->acc_x.ptr(args->rect.lo), args->acc_y.ptr(args->rect.lo),
           args->acc_z.ptr(args->rect.lo));

#ifdef INT32
    printf(" alpha = %d \n", args->alpha);
#elif DOUBLE
    printf(" alpha = %f \n", args->alpha);
#endif
  }
#endif

  Rect<1> rect;
  rect.lo = args->rect.lo;
  rect.hi = args->rect.hi;

  unsigned int index = rect.lo.value;
  unsigned int subregion_size = WIDTH*HEIGHT/(NUM_SUBREGIONS * NUM_SUBREGIONS);
  unsigned int subregion_width = WIDTH/NUM_SUBREGIONS;
  unsigned int subregion_height = HEIGHT/NUM_SUBREGIONS;
  unsigned int subregion_index = index/subregion_size;
  unsigned int start_row = (subregion_index/NUM_SUBREGIONS) * subregion_height;
  unsigned int start_col = (subregion_index%NUM_SUBREGIONS) * subregion_width;
  unsigned int end_row = start_row + subregion_height - 1;
  unsigned int end_col = start_col + subregion_width - 1;



  AccessorRO block_acc_y;
  AccessorRO block_acc_x;
  AccessorWD block_acc_z;

  // set base pointer for the new block accessors
  block_acc_x.accessor.base = (uintptr_t)mem_alloc(BLOCK_SIZE * WIDTH * sizeof(TYPE));
  block_acc_y.accessor.base = (uintptr_t)mem_alloc(BLOCK_SIZE * WIDTH * sizeof(TYPE));
  block_acc_z.accessor.base = (uintptr_t)mem_alloc(sizeof(TYPE));
  // set strides from base accessor
  block_acc_x.accessor.strides = args->acc_x.accessor.strides;
  block_acc_y.accessor.strides = args->acc_y.accessor.strides;
  block_acc_z.accessor.strides = args->acc_z.accessor.strides;

  // iterator through all elements

  // int counter = 0;

  unsigned int curr_row = start_row + tasklet_id/subregion_width;
  unsigned int curr_col = tasklet_id%subregion_width;
  for(unsigned int counter = tasklet_id; counter < subregion_size; counter += NR_TASKLETS){
    //read data
    Rect<1> temp_rect;
    temp_rect.lo = 0;
    temp_rect.hi = WIDTH*HEIGHT;
    Legion::PointInRectIterator<1> pir_a(temp_rect);
    Legion::PointInRectIterator<1> pir_b(temp_rect);
    Legion::PointInRectIterator<1> pir_z(rect);
    pir_a += curr_row*WIDTH;
    pir_b += (curr_col+start_col)*WIDTH;
    READ_BLOCK(*pir_a, args->acc_x, block_acc_x, WIDTH * BLOCK_SIZE* sizeof(TYPE));
    READ_BLOCK(*pir_b, args->acc_y, block_acc_y, WIDTH * BLOCK_SIZE* sizeof(TYPE));

    //calculation
    Rect<1> block_rect;
    block_rect.lo = 0;
    block_rect.hi = WIDTH-1;
    TYPE sum=0;
    for (Legion::PointInRectIterator<1> pir_block(block_rect); pir_block();
         pir_block++) {
      // printf("(%f, %f) ", block_acc_x[*pir] ,block_acc_y[*pir]);

      sum += block_acc_x[*pir_block] * block_acc_y[*pir_block];
      // block_acc_z.write(*pir_block, args->alpha * block_acc_x[*pir_block] +
      //                                   block_acc_y[*pir_block]);
    }

    //write data into temp block
    block_rect.lo = 0;
    block_rect.hi = 0;
    Legion::PointInRectIterator<1> pir_block(block_rect);
    block_acc_z.write(*pir_block, sum);

    //write data into accessor
    pir_z+=counter;
    WRITE_BLOCK(*pir_z, args->acc_z, block_acc_z, sizeof(TYPE));


    curr_row += NR_TASKLETS/subregion_width;
    curr_col += NR_TASKLETS;
    curr_col %= subregion_width;
  }
  // unsigned int index = tasklet_id;

  // unsigned int total_ele = WIDTH * HEIGHT;

  // for(; index<total_ele; index +=NR_TASKLETS){

  //   unsigned int row = index/WIDTH;
  //   unsigned int col = index%WIDTH;
  //   Legion::PointInRectIterator<1> pir_a(rect);
  //   Legion::PointInRectIterator<1> pir_b(rect);
  //   Legion::PointInRectIterator<1> pir_z(rect);

  //   pir_a += row*WIDTH;
  //   READ_BLOCK(*pir_a, args->acc_x, block_acc_x, WIDTH * BLOCK_SIZE* sizeof(TYPE));

  //   pir_b += col*WIDTH;
  //   READ_BLOCK(*pir_b, args->acc_y, block_acc_y, WIDTH * BLOCK_SIZE* sizeof(TYPE));

  //   Rect<1> block_rect;
  //   block_rect.lo = 0;
  //   block_rect.hi = WIDTH-1;
  //   TYPE sum=0;

  //   for (Legion::PointInRectIterator<1> pir_block(block_rect); pir_block();
  //        pir_block++) {
  //     // printf("(%f, %f) ", block_acc_x[*pir] ,block_acc_y[*pir]);

  //     sum += block_acc_x[*pir_block] * block_acc_y[*pir_block];
  //     // block_acc_z.write(*pir_block, args->alpha * block_acc_x[*pir_block] +
  //     //                                   block_acc_y[*pir_block]);
  //   }

  //   // printf("sum of tasklet \n");
  //   block_rect.lo = 0;
  //   block_rect.hi = 0;
  //   Legion::PointInRectIterator<1> pir_block(block_rect);
  //   block_acc_z.write(*pir_block, sum);

  //   pir_z += index;

  //   WRITE_BLOCK(*pir_z, args->acc_z, block_acc_z, sizeof(TYPE));
  // }

  // for (Legion::PointInRectIterator<1> pir(rect); pir();
  //      pir += (NR_TASKLETS * WIDTH * BLOCK_SIZE)) {

  //   // read blocks to respective base pointers
  //   // #define READ_BLOCK(point, acc_full, acc_block, bytes)  

  //   READ_BLOCK(*pir, args->acc_x, block_acc_x, WIDTH * BLOCK_SIZE* sizeof(TYPE));
  //   READ_BLOCK(*pir, args->acc_y, block_acc_y, WIDTH * BLOCK_SIZE* sizeof(TYPE));
  //   READ_BLOCK(*pir, args->acc_z, block_acc_z, WIDTH * BLOCK_SIZE* sizeof(TYPE));

  //   Rect<1> block_rect;
  //   block_rect.lo = 0;
  //   block_rect.hi = WIDTH-1;

  //   TYPE sum=0;
  //   // block iterator
  //   for (Legion::PointInRectIterator<1> pir_block(block_rect); pir_block();
  //        pir_block++) {
  //     // printf("(%f, %f) ", block_acc_x[*pir] ,block_acc_y[*pir]);

  //     sum += block_acc_x[*pir_block] * block_acc_y[*pir_block];
  //     block_acc_z.write(*pir_block, args->alpha * block_acc_x[*pir_block] +
  //                                       block_acc_y[*pir_block]);
  //   }
  //   Legion::PointInRectIterator<1> pir_block(block_rect);
  //   block_acc_z.write(*pir_block, sum);
  //   // write block
  //   // #define WRITE_BLOCK(point, acc_full, acc_block, bytes)
  //   WRITE_BLOCK(*pir, args->acc_z, block_acc_z, WIDTH * BLOCK_SIZE * sizeof(TYPE));

  //   counter++;
  // }
  return 0;
}