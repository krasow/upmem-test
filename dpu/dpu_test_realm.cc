extern "C" {
#include <stdint.h>
#include <defs.h>
#include <mram.h>
#include <alloc.h>
#include <barrier.h>
}

#include <realm/upmem/realm_c_upmem.h>

typedef enum DPU_LAUNCH_KERNELS{
  test,
  nr_kernels = 1
} DPU_LAUNCH_KERNELS;

typedef struct __DPU_LAUNCH_ARGS {
    Rect<2> bounds;
    AffineAccessor<int, 2> linear_accessor;
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
    if (tasklet_id == 0) {
        printf("running with %d Tasklets\n", NR_TASKLETS);
    }

    unsigned int row, column;
    row = 256;
    column = 256;

    int res = 0;

    // tasklets need to be a multiple of row
    for(unsigned int idx = tasklet_id; idx < row; idx += NR_TASKLETS){
        for(unsigned int idy = 0; idy < column; idy++){
            res = args->linear_accessor[Point<2>(idx, idy)];
            assert(res == 5);
        }
    }  

    for(unsigned int idx = tasklet_id; idx < row; idx += NR_TASKLETS){
        for(unsigned int idy = 0; idy < column; idy++){
            // interchanging idy and idx will result in wrong results
            // TODO: can fix by adding reader locks
            args->linear_accessor.write(Point<2>(idy, idx), idx*column + idy);
            res = args->linear_accessor[Point<2>(idy, idx)];
            assert(res == (idx*column + idy));
        }
    }  

    return 0;
}