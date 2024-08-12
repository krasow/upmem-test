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


// typedef FieldAccessor<READ_ONLY,double,1,coord_t,
//                       Realm::AffineAccessor<double,1,coord_t> > AccessorRO;
// typedef FieldAccessor<WRITE_DISCARD,double,1,coord_t,
//                       Realm::AffineAccessor<double,1,coord_t> > AccessorWD;


typedef struct __DPU_LAUNCH_ARGS {
//   Rect<1> rect;
//   AccessorRO acc_y;
//   AccessorRO acc_x;
//   AccessorWD acc_z;
  AffineAccessor<double, 1> acc_y;
  AffineAccessor<double, 1> acc_x;
  AffineAccessor<double, 1> acc_z;
  double alpha;
  DPU_LAUNCH_KERNELS kernel;
  PADDING(8);
} __attribute__((aligned(8))) __DPU_LAUNCH_ARGS;


typedef struct DPU_LAUNCH_ARGS {
    char paddd[128];
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

    // for(unsigned int idx; idx < 1024; idx++) {
    //     double res = args->alpha * args->acc_x[Point<1>(idx)] + args->acc_y[Point<1>(idx)];
    //     args->acc_z.write(Point<1>(idx), res);                    
    // }
                  
    // for (PointInRectIterator<1> pir(args->rect); pir(); pir++) 
    //     args->acc_z[*pir] = args->alpha * args->acc_x[*pir] + args->acc_y[*pir]; 

    return 0;
}