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

#ifndef _COMMON_H_
#define _COMMON_H_

#define USE_LEGION

/* Brings in headers to define accessors */
#include <realm/upmem/upmem_common.h>

typedef FieldAccessor<LEGION_READ_ONLY,TYPE,1,coord_t,
                      Realm::AffineAccessor<TYPE,1,coord_t> > AccessorRO;
typedef FieldAccessor<LEGION_WRITE_DISCARD,TYPE,1,coord_t,
                      Realm::AffineAccessor<TYPE,1,coord_t> > AccessorWD;


typedef enum DPU_LAUNCH_KERNELS{
  test,
  nr_kernels = 1
} DPU_LAUNCH_KERNELS;

//alpha will stand for the bins
typedef struct DPU_LAUNCH_ARGS{
  TYPE bins;
  TYPE depth;
  Rect<1> rect;
  Rect<1> rect_y;
  AccessorWD acc_y;
  AccessorRO acc_x;
  DPU_LAUNCH_KERNELS kernel;
  PADDING(8);
} __attribute__((aligned(8))) DPU_LAUNCH_ARGS;

#endif