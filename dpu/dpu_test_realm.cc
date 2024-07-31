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
    printf("%p\n", DPU_MRAM_HEAP_POINTER);

    somethingelse* test = (somethingelse*)(&ARGS);
    
    unsigned int i, j;
    unsigned int row, column;

    row = 1000;
    column = 1000;

    // for (i = 0; i < row; i++) {
    //     for (j = 0; j < column; j++) {
    //         ok = test->linear_accessor[Point<2>(i, j)];
    //         assert(ok == 5.0000);
    //     }
    // }  

    // ok = test->linear_accessor[Point<2>(-1, 0)];

    // printf("value %lf\n", ok);
    // double *buff = (double *)mem_alloc(row*column*sizeof(double));


    // mram_read((__mram_ptr void const *)((uintptr_t)DPU_MRAM_HEAP_POINTER),
    //     (void *)(buff), row*column*sizeof(double));


    // for (i = 0; i < row; i++) {
    //     for (j = 0; j < column; j++) {
    //         ok = buff[i*column + j];
    //         printf("%f ", ok);
    //         assert(ok == 5.0000);
    //     }
    //     printf("\n"); 
    // }  
    
    printf("ok is %1.3f at (%d, %d)\n", ok, row, column);


    // for (i = 0; i <= row; i++) {
    //     for (j = 0; j <= column; j++) {
    //         test->linear_accessor.write(Point<2>(i, j), 3.32);
    //         ok = test->linear_accessor[Point<2>(i, j)];
    //         assert(ok == 3.32);
    //     }
    // }  

    ok = test->linear_accessor[Point<2>(row, column)];
    printf("ok is %1.3f at (%d, %d) after write\n", ok, row, column);

    return 0;
}