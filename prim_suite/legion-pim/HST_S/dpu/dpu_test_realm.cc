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

#define BLOCK_SIZE 32

typedef struct __DPU_LAUNCH_ARGS {
  char paddd[256];
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
    printf("DEVICE:::: Running HST computation with xptr %p",
           args->acc_x.ptr(args->rect.lo));

    printf("DEVICE::: my tasklet id is %d, the lower bound of the rect is %d \n", tasklet_id, args->rect_y.lo.value);
  }
#endif

  Rect<1> rect;
  rect.lo = args->rect.lo + tasklet_id * BLOCK_SIZE;
  rect.hi = args->rect.hi;

  AccessorWD block_acc_y;
  AccessorRO block_acc_x;

  // set base pointer for the new block accessors
  block_acc_x.accessor.base = (uintptr_t)mem_alloc((BLOCK_SIZE) * sizeof(TYPE));
  block_acc_y.accessor.base = (uintptr_t)mem_alloc((args->bins) * sizeof(TYPE));
  // set strides from base accessor
  block_acc_x.accessor.strides = args->acc_x.accessor.strides;
  block_acc_y.accessor.strides = args->acc_y.accessor.strides;


// #ifdef PRINT_UPMEM
//   Rect<1> temp_rect;
//   temp_rect.lo = 0;
//   temp_rect.hi = 256 - 1;
//   unsigned int range = temp_rect.hi.value - temp_rect.lo.value + 1;
//   Legion::PointInRectIterator<1> pir_a(temp_rect);
//   READ_BLOCK(*pir_a, args->acc_x, block_acc_x, 256 * sizeof(TYPE));
//   unsigned int counter = 0;
//   Rect<1> block_rect;
//   block_rect.lo = 0;
//   block_rect.hi = 256-1;
//   for (Legion::PointInRectIterator<1> pir_block(block_rect); pir_block(); pir_block++) {
//     if(counter%16==0) printf("\n");
//     printf("%f ", block_acc_x[*pir_block]);
//     counter++;
//   }
// #endif

  Rect<1> output_rect;
  output_rect.lo = args->rect_y.lo;
  output_rect.hi = args->rect_y.lo+256;
  Legion::PointInRectIterator<1> output_pir(output_rect);
  READ_BLOCK(*output_pir, args->acc_y, block_acc_y, args->bins * sizeof(TYPE));


  // iterate through all elements
  for (Legion::PointInRectIterator<1> pir(rect); pir();
       pir += (NR_TASKLETS * BLOCK_SIZE)) {

    // read blocks to respective base pointers
    // #define READ_BLOCK(point, acc_full, acc_block, bytes)
    READ_BLOCK(*pir, args->acc_x, block_acc_x, BLOCK_SIZE * sizeof(TYPE));
    // READ_BLOCK(*output_pir, args->acc_y, block_acc_y, args->bins * sizeof(TYPE));

    Rect<1> block_rect;
    block_rect.lo = 0;
    block_rect.hi = BLOCK_SIZE-1;

    Rect<1> bin_rect;
    bin_rect.lo = 0;
    bin_rect.hi = 256-1;

    // block iterator
    for (Legion::PointInRectIterator<1> pir_block(block_rect); pir_block();
         pir_block++) {

      TYPE curr_val = block_acc_x[*pir_block];
      int bin_index = curr_val * args->bins >> args->depth;
      Legion::PointInRectIterator<1> output_pir_block(bin_rect);
      output_pir_block += (bin_index);
      // for(int i=0; i<bin_index; i++) output_pir_block++;

// #ifdef PRINT_UPMEM
//       printf("the current value is %d and the corresponding index is %d\n", curr_val, bin_index);
// #endif


      TYPE ori_bin_val = block_acc_y[*output_pir_block];

      block_acc_y.write(*output_pir_block, ori_bin_val + 1);
    }

    // write block
    // #define WRITE_BLOCK(point, acc_full, acc_block, bytes)
    // WRITE_BLOCK(*pir, args->acc_z, block_acc_z, BLOCK_SIZE * sizeof(TYPE));
  }

#ifdef PRINT_UPMEM
  for (Legion::PointInRectIterator<1> pir_block(output_rect); pir_block(); pir_block++) {
    printf("the value is %d\n", block_acc_y[*pir_block]);
  }
  // fflush(stdout);
#endif

  WRITE_BLOCK(*output_pir, args->acc_y, block_acc_y, args->bins * sizeof(TYPE));

  return 0;
}
