# Copyright 2024 Stanford University
#  			     Los Alamos National Laboratory, Northwestern University
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#


ifndef LG_RT_DIR
$(error LG_RT_DIR variable is not defined, aborting build)
endif

# Flags for directing the runtime makefile what to include
DEBUG           ?= 1		# Include debugging symbols
MAX_DIM         ?= 3		# Maximum number of dimensions
OUTPUT_LEVEL    ?= LEVEL_INFO	# Compile time logging level
USE_CUDA        ?= 0		# Include CUDA support (requires CUDA)
USE_HIP         ?= 0		# Include HIP support (requires HIP)
HIP_TARGET      ?= ROCM
USE_GASNET      ?= 0		# Include GASNet support (requires GASNet)
USE_HDF         ?= 0		# Include HDF5 support (requires HDF5)
ALT_MAPPERS     ?= 0		# Include alternative mappers (not recommended)
USE_UPMEM		?= 1		# Include UPMEM support (requries UPMEM SDK)

# Put the binary file name here
OUTFILE		?= upmem_test
# List all the application source files here
GEN_SRC		?= host/upmem_realm_test.cc	# .cc files
GEN_GPU_SRC	?= 
GEN_UPMEM_SRC ?= dpu/dpu_test_realm.cc  # .c files for UPMEM source 


# You can modify these variables, some will be appended to by the runtime makefile
INC_FLAGS	?=
CC_FLAGS	?= 
NVCC_FLAGS	?=
HIPCC_FLAGS ?=
UPMEMCC_FLAGS ?= -DUINT32 -g0 -g
GASNET_FLAGS	?=
LD_FLAGS	?=

###########################################################################
#
#   Don't change anything below here
#   
###########################################################################

include $(LG_RT_DIR)/runtime.mk
