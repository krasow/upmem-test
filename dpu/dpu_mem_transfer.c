// https://github.com/CMU-SAFARI/prim-benchmarks/blob/main/VA/dpu/task.c


#include <defs.h>
#include <mram.h>
#include <alloc.h>

int main_kernel1();

int main(void) { 
    return main_kernel1();
}

int main_kernel1() {

    unsigned int tasklet_id = me();

    if (tasklet_id == 0){ // Initialize once the cycle counter
        mem_reset(); // Reset the heap
    }
    return 0;
}