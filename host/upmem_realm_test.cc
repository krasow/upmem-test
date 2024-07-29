#include "realm.h"
#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <thread>
#include <time.h>
#include <unistd.h>

#include <realm/upmem/upmem_access.h>


#if !defined(REALM_USE_UPMEM) 
#error Realm not compiled with UPMEM enabled
#endif // !defined(REALM_USE_UPMEM) 

using namespace Realm;

Logger log_app("app");

typedef struct {
  Rect<2> bounds;
  AffineAccessor<double, 2>  linear_accessor;
  // padding of multiple of 8 bytes
  PADDING(8);
} __attribute__((aligned(8))) __DEVICE_DPU_LAUNCH_ARGS;


// typedef struct  {
//   Upmem::Rect<2> bounds;
//   Upmem::AffineAccessor<double, 2>  linear_accessor;
// } __attribute__((aligned(8))) __DEVICE_DPU_LAUNCH_ARGS;

typedef struct {
  Rect<2> bounds;
  RegionInstance linear_instance;
} __attribute__((aligned(8))) DPUTaskArgs;


enum {
  TOP_LEVEL_TASK = Processor::TASK_ID_FIRST_AVAILABLE + 0,
  DPU_LAUNCH_TASK,
  CHECK_TASK
};

#define  DPU_LAUNCH_TASK_BINARY  "dpu/dpu_test_realm.up.o"

static void dpu_launch_task(const void *data, size_t datalen,
                            const void *userdata, size_t userlen, Processor dpu) {

  log_app.print() << "dpu launch task running on " << dpu;

  assert(datalen == sizeof(DPUTaskArgs));
  DPUTaskArgs task_args = *static_cast<const DPUTaskArgs *>(data);

  AffineAccessor<double, 2> linear_accessor(task_args.linear_instance, 0);

  {
    __DEVICE_DPU_LAUNCH_ARGS args;
    args.bounds = task_args.bounds;
    args.linear_accessor = linear_accessor;
    dpu_set_t *stream = new dpu_set_t;
    Upmem::LaunchKernel(DPU_LAUNCH_TASK_BINARY, (void**)&args, "ARGS", sizeof(__DEVICE_DPU_LAUNCH_ARGS),  stream);
  }
}

void top_level_task(const void *args, size_t arglen, const void *userdata,
                    size_t userlen, Processor dpu) {
  log_app.print() << "top task running on " << dpu;

  const size_t width = 1024, height = 1024;
  std::vector<size_t> field_sizes(1, sizeof(double));

  Rect<2> bounds(Point<2>(0, 0), Point<2>(width - 1, height - 1));

  // ==== Allocating DPU memory with Realm ====
  Memory dpu_mem = Machine::MemoryQuery(Machine::get_machine())
                       .has_capacity(bounds.volume() /* index space */ * 1 /* number of fields */ * sizeof(double) /* type of field */)
                       .best_affinity_to(dpu)
                       .first();

  assert((dpu_mem != Memory::NO_MEMORY) && "Failed to find suitable DPU memory to use!");

  // Now create a 2D instance like we normally would
  RegionInstance linear_instance = RegionInstance::NO_INST;
  Event linear_instance_ready_event =
      RegionInstance::create_instance(linear_instance, dpu_mem, bounds, field_sizes,
                                      /*SOA*/ 1, ProfilingRequestSet());

  // ==== Data Movement ====
  Event fill_done_event = Event::NO_EVENT;
  {
    std::vector<CopySrcDstField> srcs(1), dsts(1);
    // Realm does not currently support non-affine fill operations, so fill the linear
    // array with the value we want and copy the linear array to the cuda array instance,
    // making sure to chain the events as we go along
    // While we could use cuda APIs to achieve the same result (using cudaMemcpy3D or
    // cudaMemcpyToArray), we would need to synchronize the instance creation at this
    // point for the linear array, or allocate our own gpu-accessible staging buffer, or
    // have cuda do a pageable memcpy to the array.  Instead, we just let realm handle it
    // for us asynchrnously.

    // Fill the linear array with zeros.
    srcs[0].set_fill<double>(5.0f);
    dsts[0].set_field(linear_instance, 0, field_sizes[0]);
    fill_done_event =
        bounds.copy(srcs, dsts, ProfilingRequestSet(), linear_instance_ready_event);
  }


 // ==== Task Spawning ====

  Event dpu_task_done_event = Event::NO_EVENT;
  {
    DPUTaskArgs args;
    args.bounds = bounds;
    args.linear_instance = linear_instance;
    dpu_task_done_event =
        dpu.spawn(DPU_LAUNCH_TASK, &args, sizeof(args), fill_done_event);
  };

  Runtime::get_runtime().shutdown(dpu_task_done_event);
}

void check_task(const void *args, size_t arglen, const void *userdata,
                size_t userlen, Processor p) {
  log_app.print() << "check task " << p;
}

int main(int argc, char **argv) {
  Runtime rt;

  rt.init(&argc, &argv);

  Processor::register_task_by_kind(
      Processor::DPU_PROC, false /*!global*/, TOP_LEVEL_TASK,
      CodeDescriptor(top_level_task), ProfilingRequestSet(), 0, 0)
      .wait();

  Processor::register_task_by_kind(
      Processor::DPU_PROC, false /*!global*/, DPU_LAUNCH_TASK,
      CodeDescriptor(dpu_launch_task), ProfilingRequestSet(), 0, 0)
      .wait();

  // Processor::register_task_by_kind(Processor::LOC_PROC, false /*!global*/,
  //                                  CHECK_TASK, CodeDescriptor(check_task),
  //                                  ProfilingRequestSet(), 0, 0)
  //     .wait();

  Processor p = Machine::ProcessorQuery(Machine::get_machine())
                    .only_kind(Processor::DPU_PROC)
                    .first();

  assert((p != Processor::NO_PROC) && "Unable to find suitable DPU processor");
  
  
  rt.collective_spawn(p, TOP_LEVEL_TASK, 0, 0);

  return rt.wait_for_shutdown();
}