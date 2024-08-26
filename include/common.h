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

#define USE_REALM // use realm 

/* Brings in headers to define accessors */
#include <realm/upmem/upmem_common.h>

typedef enum {
  test,
  nr_kernels = 1,
} DPU_LAUNCH_KERNELS;

typedef struct DPU_LAUNCH_ARGS {
  Rect<2> bounds;
  AffineAccessor<TYPE, 2> arrayA_accessor;
  AffineAccessor<TYPE, 2> arrayB_accessor;
  AffineAccessor<TYPE, 2> arrayC_accessor;
  // which kernel to launch
  DPU_LAUNCH_KERNELS kernel;
  // padding of multiple of 8 bytes
  PADDING(8);
} __attribute__((aligned(8))) DPU_LAUNCH_ARGS;

#endif