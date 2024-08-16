extern "C" {
#include <stdint.h>
#include <defs.h>
#include <mram.h>
#include <alloc.h>
#include <barrier.h>
}

#include <realm/upmem/legion_c_upmem.h>

typedef FieldAccessor<LEGION_READ_ONLY,double,1,coord_t,
                      Realm::AffineAccessor<double,1,coord_t> > AccessorRO;
typedef FieldAccessor<LEGION_WRITE_DISCARD,double,1,coord_t,
                      Realm::AffineAccessor<double,1,coord_t> > AccessorWD;

typedef enum DPU_LAUNCH_KERNELS{
  test,
  nr_kernels = 1
} DPU_LAUNCH_KERNELS;


typedef struct __DPU_LAUNCH_ARGS {
  double alpha;
  Rect<1> rect;
  AccessorRO acc_y;
  AccessorRO acc_x;
  AccessorWD acc_z;
  DPU_LAUNCH_KERNELS kernel;
  PADDING(8);
} __attribute__((aligned(8))) __DPU_LAUNCH_ARGS;


typedef struct DPU_LAUNCH_ARGS {
    char paddd[256];
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

    printf("DEVICE:::: Running daxpy computation with alpha %.8f xptr %p, y_ptr %p, z_ptr %p...\n", 
        args->alpha,
        args->acc_x.ptr(args->rect.lo), args->acc_y.ptr(args->rect.lo), args->acc_z.ptr(args->rect.lo));

    for (Legion::PointInRectIterator<1> pir(args->rect); pir(); pir++) {
       args->acc_z.write(*pir, args->alpha * args->acc_x[*pir] + args->acc_y[*pir]); 
      //  printf("read %f,\t",args->alpha * args->acc_x[*pir] + args->acc_y[*pir]);
      //  printf("write %f\n",args->acc_z[*pir]);
    }

    return 0;
}