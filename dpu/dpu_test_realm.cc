// https://github.com/CMU-SAFARI/prim-benchmarks/blob/main/VA/dpu/task.c

extern "C" {
#include <stdint.h>
#include <defs.h>
#include <mram.h>
#include <alloc.h>
#include <barrier.h>
}

#include <realm/upmem/realm_c_upmem.h>


typedef struct somethingelse {
    Rect<2> bounds;
    AffineAccessor<double, 2> linear_accessor;
    PADDING(8);
} __attribute__((aligned(8))) somethingelse;

typedef struct __DEVICE_DPU_LAUNCH_ARGS {
    // Rect<2> bounds;
    // AffineAccessor<double, 2> linear_accessor;
    // PADDING(16);
    char paddd[48];
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


    /*
    somethingelse test; 

    printf("sizeof(somethingelse): %d\n", sizeof(somethingelse));
    printf("offset of bounds: %lu\n", offsetof(somethingelse, bounds));
    printf("offset of linear_accessor: %lu\n", offsetof(somethingelse, linear_accessor));
    printf("sizeof(test.linear_accessor): %d\n", sizeof(test.linear_accessor));
    */
    
    // __DEVICE_DPU_LAUNCH_ARGS *ARGS = (__DEVICE_DPU_LAUNCH_ARGS *)mem_alloc(sizeof(__DEVICE_DPU_LAUNCH_ARGS));

    // mram_read((__mram_ptr void const*)(DPU_MRAM_HEAP_POINTER), (void*)ARGS, sizeof(__DEVICE_DPU_LAUNCH_ARGS));
    
    // double value = test->linear_accessor[Point<2>(0, 0)];
    
    // Point<2> psomethign(33, 1);
    // assert((uint64_t)test != 0 && ((uint64_t)test % 8) == 0);

    printf("%lf \n", test->linear_accessor[Point<2>(33, 63)]);

    // for (unsigned int i = 0; i < 2; i++) {
    //     for (unsigned int j = 0; j < 20; j++) {
    //         printf("%f ", test->linear_accessor[Point<2>(i, j)]);
    //     }
    // }  


    return 0;
}