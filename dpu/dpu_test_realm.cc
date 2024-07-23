// https://github.com/CMU-SAFARI/prim-benchmarks/blob/main/VA/dpu/task.c

extern "C" {
#include <stdint.h>
#include <defs.h>
#include <mram.h>
#include <alloc.h>
#include <barrier.h>
}


#include <realm/upmem/realm_c_upmem.h>


typedef struct {
    Rect<2> bounds;
    AffineAccessor<double, 2> linear_accessor;
} __attribute__((aligned(8))) __DPU_LAUNCH_TASK_ARGS;


// __host __DPU_LAUNCH_TASK_ARGS DPU_LAUNCH_TASK_ARGS;  

int main_kernel1();

// Barrier
BARRIER_INIT(my_barrier, NR_TASKLETS);


int main(void) { 
    return main_kernel1();
}

int main_kernel1() {
    unsigned int tasklet_id = me();

    __DPU_LAUNCH_TASK_ARGS *DPU_LAUNCH_TASK_ARGS = (__DPU_LAUNCH_TASK_ARGS *)mem_alloc(sizeof(__DPU_LAUNCH_TASK_ARGS));

    // mram_read((__mram_ptr void const*)(DPU_MRAM_HEAP_POINTER), (void*)DPU_LAUNCH_TASK_ARGS, sizeof(__DPU_LAUNCH_TASK_ARGS));
    
    // double value = DPU_LAUNCH_TASK_ARGS->linear_accessor[Point<2>(0, 0)];

    // return (int) value;


    return 0;
}