/**
 * \file realm_defines.h
 * Public-facing definitions of variables configured at build time
 * DO NOT EDIT THIS FILE WITHOUT CHANGING runtime.mk ALSO
 */

// ******************** IMPORTANT **************************
//
// This file is PURE C, **NOT** C++. Keep any C++-isms in
// legion_types.h, or elsewhere.
//
// ******************** IMPORTANT **************************

#define REALM_VERSION "legion-23.12.0-6701-gd793de769-dirty"

#define DEBUG_REALM

#define REALM_LIMIT_SYMBOL_VISIBILITY

#define COMPILE_TIME_MIN_LEVEL LEVEL_INFO

#define REALM_MAX_DIM 3

/* #undef REALM_USE_OPENMP */
/* #undef REALM_OPENMP_SYSTEM_RUNTIME */
/* #undef REALM_OPENMP_GOMP_SUPPORT */
/* #undef REALM_OPENMP_KMP_SUPPORT */

/* #undef REALM_USE_PYTHON */

/* #undef REALM_USE_CUDA */
/* #undef REALM_USE_CUDART_HIJACK */
/* #undef REALM_CUDA_DYNAMIC_LOAD */

/* #undef REALM_USE_HIP */
/* #undef REALM_USE_HIP_HIJACK */

/* #undef REALM_USE_KOKKOS */

/* #undef REALM_USE_GASNET1 */
/* #undef REALM_USE_GASNETEX */

/* technically these are defined by per-conduit GASNet include files,
 * but we do it here as well for the benefit of applications that care
 */
/* #undef GASNET_CONDUIT_MPI */
/* #undef GASNET_CONDUIT_IBV */
/* #undef GASNET_CONDUIT_UDP */
/* #undef GASNET_CONDUIT_ARIES */
/* #undef GASNET_CONDUIT_GEMINI */
/* #undef GASNET_CONDUIT_PSM */
/* #undef GASNET_CONDUIT_UCX */
/* #undef GASNET_CONDUIT_OFI */

/* #undef REALM_USE_MPI */
/* #undef REALM_MPI_HAS_COMM_SPLIT_TYPE */

/* #undef REALM_USE_UCX */
/* #undef REALM_UCX_DYNAMIC_LOAD */

/* #undef REALM_USE_LLVM */
/* #undef REALM_ALLOW_MISSING_LLVM_LIBS */

/* #undef REALM_USE_HDF5 */

#define REALM_USE_LIBDL
/* #undef REALM_USE_DLMOPEN */

/* #undef REALM_USE_HWLOC */

/* #undef REALM_USE_PAPI */

/* #undef REALM_USE_NVTX */

#define REALM_USE_UNWIND

/* #undef REALM_USE_LIBDW */

/* #undef REALM_USE_SHM */
/* #undef REALM_HAS_POSIX_FALLOCATE64 */

/* #undef REALM_RESPONSIVE_TIMELIMIT */

/* #undef REALM_USE_SIMPLE_TEST */
#define REALM_USE_UPMEM
