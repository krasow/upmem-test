// https://github.com/CMU-SAFARI/prim-benchmarks/blob/main/VA/dpu/task.c

extern "C" {
#include <stdint.h>
#include <defs.h>
#include <mram.h>
#include <alloc.h>
#include <barrier.h>
}

#include <realm/upmem/realm_c_upmem.h>

volatile double ok = 0.0;

typedef struct somethingelse {
    Rect<2> bounds;
    AffineAccessor<double, 2> linear_accessor;
    PADDING(8);
} __attribute__((aligned(8))) somethingelse;

typedef struct __DEVICE_DPU_LAUNCH_ARGS {
    // Rect<2> bounds;
    // AffineAccessor<double, 2> linear_accessor;
    // PADDING(16);
    char paddd[64];
} __attribute__((aligned(8))) __DEVICE_DPU_LAUNCH_ARGS;

__host __DEVICE_DPU_LAUNCH_ARGS ARGS;  

int main_kernel1();

// Barrier
BARRIER_INIT(my_barrier, NR_TASKLETS);


int main(void) { 
    return main_kernel1();
}

int main_kernel1() {
    unsigned int tasklet_id = me();

    somethingelse* test = (somethingelse*)(&ARGS);
    
    printf("test->linear_accessor.strides[0]: %lu\n", test->linear_accessor.strides[0]);
    printf("test->linear_accessor.strides[1]: %lu\n", test->linear_accessor.strides[1]);
    
    unsigned int i, j;
    for (i = 0; i < 100; i++) {
        for (j = 0; j < 100; j++) {
            ok = test->linear_accessor[Point<2>(i, j)];
        }
    }  

    printf("ok is %1.3f at (%d, %d)\n", ok, 99, 99);


    for (i = 0; i < 100; i++) {
        for (j = 0; j < 100; j++) {
            test->linear_accessor.write(Point<2>(i, j), 3.32);
        }
    }  

    ok = test->linear_accessor[Point<2>(99, 99)];
    printf("ok is %1.3f at (%d, %d) after write\n", ok, 99, 99);

    return 0;
}