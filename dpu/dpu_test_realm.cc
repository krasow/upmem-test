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
  char paddd[1024];
} __attribute__((aligned(8))) __DPU_LAUNCH_ARGS;

__host __DPU_LAUNCH_ARGS ARGS;

DPU_LAUNCH_ARGS *args = (DPU_LAUNCH_ARGS *)(&ARGS);

int main_kernel1();

// Barrier
BARRIER_INIT(my_barrier, NR_TASKLETS);

int (*kernels[nr_kernels])(void) = {main_kernel1};

int main(void) { return kernels[args->kernel](); }

int main_kernel1() {
  // unsigned int tasklet_id = me();

#ifdef PRINT_UPMEM
  // printf("DEVICE::: my tasklet id is %d, the lower bound of the rect is %d and %d\n", tasklet_id, args->rect.lo.values[0], args->rect.lo.values[1]);
  // printf("the low value of rect_x is %d and %d and the high value of rect_x is %d and %d\n", args->rect_x.lo.values[0], args->rect_x.lo.values[1], args->rect_x.hi.values[0], args->rect_x.hi.values[1]);
  // printf("the low value of rect_y is %d and %d and the high value of rect_y is %d and %d\n", args->rect_y.lo.values[0], args->rect_y.lo.values[1], args->rect_y.hi.values[0], args->rect_y.hi.values[1]);
  // printf("the number of subregions is %d\n", args->num_subregions);

#endif

//   Rect<2> rect;
//   //TODO: can i just use rect.lo = args->args.lo
//   rect.lo.values[0] = args->rect.lo.values[0];
//   rect.lo.values[1] = args->rect.lo.values[1];

//   rect.hi.values[0] = args->rect.hi.values[0];
//   rect.hi.values[1] = args->rect.hi.values[1];

//   const int w = args->w;
//   const int num_subregions = args->num_subregions;


//   unsigned int range = (rect.hi.values[0] - rect.lo.values[0] + 1) * (rect.hi.values[1] - rect.lo.values[1] + 1);
//   unsigned int subregion_width = w/num_subregions;
//   unsigned int subregion_height = w/num_subregions;
//   unsigned int start_row = args->rect_x.lo.values[0]/w;
//   unsigned int start_col = args->rect_y.lo.values[0]/w;

//   AccessorRO block_acc_y;
//   AccessorRO block_acc_x;
//   AccessorWD block_acc_z;
//   block_acc_x.accessor.base = (uintptr_t)mem_alloc((BLOCK_SIZE * w + 1) * sizeof(TYPE));
//   block_acc_y.accessor.base = (uintptr_t)mem_alloc((BLOCK_SIZE * w + 1) * sizeof(TYPE));
//   block_acc_z.accessor.base = (uintptr_t)mem_alloc(sizeof(TYPE));
//   // set strides from base accessor
//   block_acc_x.accessor.strides = args->acc_x.accessor.strides;
//   block_acc_y.accessor.strides = args->acc_y.accessor.strides;
//   block_acc_z.accessor.strides = args->acc_z.accessor.strides;

//   unsigned int curr_row = 0;
//   unsigned int curr_col = 0;

//   // iterator through all elements
//   for(unsigned int counter = tasklet_id; counter < range; counter += NR_TASKLETS){
//     //read data
//     // Rect<2> temp_rect;
//     // temp_rect.lo.values[0] = 0;
//     // temp_rect.lo.values[1] = 0;
//     // temp_rect.hi.values[0] = w*w*num_subregions;
//     Legion::PointInRectIterator<2> pir_a(args->rect_x);
//     Legion::PointInRectIterator<2> pir_b(args->rect_y);
//     Legion::PointInRectIterator<2> pir_z(rect);
//     curr_row = counter/subregion_width + start_row;
//     curr_col = counter%subregion_width+ start_col;

//     pir_a += (counter/subregion_width)*w;
//     pir_b += (counter%subregion_width)*w;
//     READ_BLOCK(*pir_a, args->acc_x, block_acc_x, w * BLOCK_SIZE* sizeof(TYPE));
//     READ_BLOCK(*pir_b, args->acc_y, block_acc_y, w * BLOCK_SIZE* sizeof(TYPE));

//     //calculation
//     Rect<2> block_rect;
//     block_rect.lo.values[0] = 0;
//     block_rect.lo.values[1] = 0;
//     block_rect.hi.values[1] = 0;
//     block_rect.hi.values[1] = w-1;
//     // block_rect.hi = w-1;
//     TYPE sum=0;
//     // printf("the subregion size")
//     printf("the current row is %u and the current col is %u \n", curr_row, curr_col);
//     for (Legion::PointInRectIterator<2> pir_block(block_rect); pir_block();
//          pir_block++) {
// #ifdef PRINT_UPMEM
//       printf("(%f, %f) ", block_acc_x[*pir_block] ,block_acc_y[*pir_block]);
// #endif
//       sum += block_acc_x[*pir_block] * block_acc_y[*pir_block];
//     }


// #ifdef PRINT_UPMEM
//     printf("\n the sum is %f\n", sum);
// #endif
//     //write data into temp block
//     Rect<2> write_rect;
//     write_rect.lo.values[0] = 0;
//     write_rect.lo.values[1] = 0;
//     write_rect.hi.values[0] = 0;
//     write_rect.hi.values[1] = 0;
//     // block_rect.lo = 0;
//     // block_rect.hi = 0;
//     Legion::PointInRectIterator<2> pir_block(write_rect);
//     block_acc_z.write(*pir_block, sum);

//     pir_z+=counter;
//     WRITE_BLOCK(*pir_z, args->acc_z, block_acc_z, sizeof(TYPE));
//   }

  
  return 0;
}
