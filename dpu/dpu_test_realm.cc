extern "C" {
#include <stdint.h>
#include <defs.h>
#include <mram.h>
#include <alloc.h>
#include <barrier.h>
}

#include <realm/upmem/realm_c_upmem.h>

volatile double ok = 0.0;

typedef enum DPU_LAUNCH_KERNELS{
  test,
  nr_kernels = 1
} DPU_LAUNCH_KERNELS;

typedef struct __DPU_LAUNCH_ARGS {
    Rect<2> bounds;
    AffineAccessor<double, 2> linear_accessor;
    DPU_LAUNCH_KERNELS kernel;
    PADDING(8);
} __attribute__((aligned(8))) __DPU_LAUNCH_ARGS;

typedef struct DPU_LAUNCH_ARGS {
    char paddd[64];
} __attribute__((aligned(8))) DPU_LAUNCH_ARGS;

__host DPU_LAUNCH_ARGS ARGS;  

__DPU_LAUNCH_ARGS* args = (__DPU_LAUNCH_ARGS*)(&ARGS);

int main_kernel1();

// Barrier
BARRIER_INIT(my_barrier, NR_TASKLETS);

int (*kernels[nr_kernels])(void) = {main_kernel1};

int main(void) { 
    return kernels[args->kernel](); 
}

int main_kernel1() {
    unsigned int tasklet_id = me();

    unsigned int i, j;
    unsigned int row, column;

    row = 500;
    column = 500;

    for (i = 0; i < row; i++) {
        for (j = 0; j < column; j++) {
            ok = args->linear_accessor[Point<2>(i, j)];
            assert(ok == 5.0000);
        }
    }  

    printf("ok is %1.3f at (%d, %d)\n", ok, row, column);


    for (i = 0; i <= row; i++) {
        for (j = 0; j <= column; j++) {
            args->linear_accessor.write(Point<2>(i, j), 3.32);
            ok = args->linear_accessor[Point<2>(i, j)];
            assert(ok == 3.32);
        }
    }  

    ok = args->linear_accessor[Point<2>(row, column)];
    printf("ok is %1.3f at (%d, %d) after write\n", ok, row, column);

    return 0;
}