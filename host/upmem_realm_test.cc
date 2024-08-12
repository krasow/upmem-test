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

#define WIDTH 32
#define HEIGHT 32

Logger log_app("app");

// for realm
enum {
  TOP_LEVEL_TASK = Processor::TASK_ID_FIRST_AVAILABLE + 0,
  DPU_LAUNCH_TASK,
  CPU_LAUNCH_TASK,
  FILL_TASK,
  CHECK_TASK
};

typedef struct {
  Rect<2> bounds;
  RegionInstance arrayA_instance;
  RegionInstance arrayB_instance;
  RegionInstance arrayC_instance;
  Upmem::Kernel kernel;
} __attribute__((aligned(8))) DPU_TASK_ARGS;

// for the device
#define DPU_LAUNCH_BINARY "dpu/dpu_test_realm.up.o"

typedef enum {
  test,
  nr_kernels = 1,
} DPU_LAUNCH_KERNELS;

typedef struct {
  Rect<2> bounds;
  AffineAccessor<int, 2> arrayA_accessor;
  AffineAccessor<int, 2> arrayB_accessor;
  AffineAccessor<int, 2> arrayC_accessor;
  // which kernel to launch
  DPU_LAUNCH_KERNELS kernel;
  // padding of multiple of 8 bytes
  PADDING(8);
} __attribute__((aligned(8))) DPU_LAUNCH_ARGS;

static void dpu_launch_task(const void *data, size_t datalen,
                            const void *userdata, size_t userlen,
                            Processor dpu) {

  log_app.print() << "dpu launch task running on " << dpu;

  assert(datalen == sizeof(DPU_TASK_ARGS));
  DPU_TASK_ARGS task_args = *static_cast<const DPU_TASK_ARGS *>(data);

  AffineAccessor<int, 2> arrayA_accessor(task_args.arrayA_instance, 0);
  AffineAccessor<int, 2> arrayB_accessor(task_args.arrayB_instance, 0);
  AffineAccessor<int, 2> arrayC_accessor(task_args.arrayC_instance, 0);

  {
    DPU_LAUNCH_ARGS args;
    args.bounds = task_args.bounds;
    args.arrayA_accessor = arrayA_accessor;
    args.arrayB_accessor = arrayB_accessor;
    args.arrayC_accessor = arrayC_accessor;
    args.kernel = test;
    // launch specific upmem kernel
    task_args.kernel.launch((void **)&args, "ARGS", sizeof(DPU_LAUNCH_ARGS));
  }
}

static void cpu_launch_task(const void *data, size_t datalen,
                            const void *userdata, size_t userlen,
                            Processor cpu) {

  log_app.print() << "cpu launch task running on " << cpu;

  assert(datalen == sizeof(DPU_TASK_ARGS));
  DPU_TASK_ARGS task_args = *static_cast<const DPU_TASK_ARGS *>(data);

  AffineAccessor<int, 2> arrayA_accessor(task_args.arrayA_instance, 0);
  AffineAccessor<int, 2> arrayB_accessor(task_args.arrayB_instance, 0);
  AffineAccessor<int, 2> arrayC_accessor(task_args.arrayC_instance, 0);

  for (uint32_t idx = 0; idx < HEIGHT; idx++) {
    for (uint32_t idy = 0; idy < WIDTH; idy++) {
      uint32_t sum = 0;

      for (uint32_t k = 0; k < WIDTH; k++) {
        uint32_t a = arrayA_accessor[Point<2>(idx, k)];
        uint32_t b = arrayB_accessor[Point<2>(k, idy)];
        sum += a * b;
      }
      arrayC_accessor.write(Point<2>(idx, idy), sum);
    }
  }
}


struct FillTaskArgs {
  RegionInstance arrayA_instance;
  RegionInstance arrayB_instance;
};

static void fill_task(const void *data, size_t datalen, const void *userdata,
                       size_t userlen, Processor p) {
  assert(datalen == sizeof(FillTaskArgs));
  const FillTaskArgs &task_args = *static_cast<const FillTaskArgs *>(data);

  AffineAccessor<int, 2> arrayA_accessor(task_args.arrayA_instance, 0);
  AffineAccessor<int, 2> arrayB_accessor(task_args.arrayB_instance, 0);

  const Rect<2> bounds =
      task_args.arrayA_instance.get_indexspace<2>().bounds;

  for (PointInRectIterator<2> pir(bounds); pir.valid; pir.step()) {
    arrayA_accessor.write(pir.p, rand() % 100);
    arrayB_accessor.write(pir.p, rand() % 100);
  }
}

struct CheckTaskArgs {
  RegionInstance host_check_instance;
  RegionInstance device_check_instance;
};

// This is the task entry point for checking the results match the expected
// value.
static void check_task(const void *data, size_t datalen, const void *userdata,
                       size_t userlen, Processor p) {
  assert(datalen == sizeof(CheckTaskArgs));
  const CheckTaskArgs &task_args = *static_cast<const CheckTaskArgs *>(data);
  AffineAccessor<int, 2> device_linear_accessor(task_args.device_check_instance, 0);
  AffineAccessor<int, 2> host_linear_accessor(task_args.host_check_instance, 0);
  
  // bounds are the same in both cases
  const Rect<2> bounds =
        task_args.device_check_instance.get_indexspace<2>().bounds;

  {
    FILE *file = fopen("array.bin", "wb");
    assert(file != NULL && "error open file");
    for (PointInRectIterator<2> pir(bounds); pir.valid; pir.step()) {
      int value = device_linear_accessor[pir.p];
      fwrite(&value, sizeof(int), 1, file);
    }
    fclose(file);
  };
  {
    FILE *file = fopen("host_array.bin", "wb");
    assert(file != NULL && "error open file");
    for (PointInRectIterator<2> pir(bounds); pir.valid; pir.step()) {
      int value = host_linear_accessor[pir.p];
      fwrite(&value, sizeof(int), 1, file);
    }
    fclose(file);
  };

  for (PointInRectIterator<2> pir(bounds); pir.valid; pir.step()) {
    int correct = host_linear_accessor[pir.p];
    int device_check = device_linear_accessor[pir.p];
    assert(correct == device_check && "ERROR: device does not equal host");
  }

}

void top_level_task(const void *args, size_t arglen, const void *userdata,
                    size_t userlen, Processor dpu) {
  log_app.print() << "top task running on " << dpu;

  // create a stream (aka dpu_set_t)
  dpu_set_t *stream = new dpu_set_t;
  // must associate a kernel with a stream
  Upmem::Kernel kern = Upmem::Kernel(DPU_LAUNCH_BINARY, stream);
  // the binary needs to be loaded before any memory operations
  kern.load();

  const size_t width = WIDTH, height = HEIGHT;
  std::vector<size_t> field_sizes(1, sizeof(int));

  Rect<2> bounds(Point<2>(0, 0), Point<2>(width - 1, height - 1));


   Processor cpu = Machine::ProcessorQuery(Machine::get_machine())
                      .only_kind(Processor::LOC_PROC)
                      .local_address_space()
                      .first();
  assert((cpu != Processor::NO_PROC) &&
         "Failed to find suitable CPU processor to check results!");

  Memory cpu_mem = Machine::MemoryQuery(Machine::get_machine())
                       .has_capacity(4 * width * height * sizeof(int))
                       .has_affinity_to(cpu)
                       .has_affinity_to(dpu)
                       .first();

  assert((cpu_mem != Memory::NO_MEMORY) &&
         "Failed to find suitable CPU memory to use!");

  std::cout << "Choosing CPU memory type " << cpu_mem.kind()
            << " for CPU processor " << cpu << std::endl;

  RegionInstance host_arrayA_instance = RegionInstance::NO_INST;
  Event host_arrayA_instance_ready =
      RegionInstance::create_instance(host_arrayA_instance, cpu_mem, bounds,
                                      field_sizes, 1, ProfilingRequestSet());

  RegionInstance host_arrayB_instance = RegionInstance::NO_INST;
  Event host_arrayB_instance_ready =
      RegionInstance::create_instance(host_arrayB_instance, cpu_mem, bounds,
                                      field_sizes, 1, ProfilingRequestSet());

  RegionInstance host_arrayC_instance = RegionInstance::NO_INST;
  Event host_arrayC_instance_ready =
      RegionInstance::create_instance(host_arrayC_instance, cpu_mem, bounds,
                                      field_sizes, 1, ProfilingRequestSet());

  // ==== Data Movement ====
  Event host_arrayA_fill_done = Event::NO_EVENT;
  {
    std::vector<CopySrcDstField> srcs(1), dsts(1);

    srcs[0].set_fill<int>(1);
    dsts[0].set_field(host_arrayA_instance, 0, field_sizes[0]);
    host_arrayA_fill_done = bounds.copy(srcs, dsts, ProfilingRequestSet(),
                                        host_arrayA_instance_ready);

  }
  Event host_arrayB_fill_done = Event::NO_EVENT;
  {
    std::vector<CopySrcDstField> srcs(1), dsts(1);

    // Fill the linear array with ones.
    srcs[0].set_fill<int>(1);
    dsts[0].set_field(host_arrayB_instance, 0, field_sizes[0]);
    host_arrayB_fill_done = bounds.copy(srcs, dsts, ProfilingRequestSet(),
                                        host_arrayB_instance_ready);
  }

  Event host_arrayC_fill_done = Event::NO_EVENT;
  {
    std::vector<CopySrcDstField> srcs(1), dsts(1);

    // Fill the linear array with ones.
    srcs[0].set_fill<int>(1);
    dsts[0].set_field(host_arrayC_instance, 0, field_sizes[0]);
    host_arrayC_fill_done = bounds.copy(srcs, dsts, ProfilingRequestSet(),
                                        host_arrayC_instance_ready);
  }
  Event host_fill_done_event = Event::merge_events(
      host_arrayA_fill_done, host_arrayB_fill_done, host_arrayC_fill_done);


  Event fill_task_done_event = Event::NO_EVENT;
  {
    FillTaskArgs args;
    args.arrayA_instance = host_arrayA_instance;
    args.arrayB_instance = host_arrayB_instance;
    fill_task_done_event =
        cpu.spawn(FILL_TASK, &args, sizeof(args), host_fill_done_event);
  }

  // ==== Allocating DPU memory with Realm ====
  Memory dpu_mem = Machine::MemoryQuery(Machine::get_machine())
                       .has_capacity(bounds.volume() /* index space */ *
                                     3 /* number of fields */ *
                                     sizeof(int) /* type of field */)
                       .best_affinity_to(dpu)
                       .first();

  assert((dpu_mem != Memory::NO_MEMORY) &&
         "Failed to find suitable DPU memory to use for bound 3!");

  // Now create a 2D instance like we normally would
  RegionInstance arrayA_instance = RegionInstance::NO_INST;
  Event arrayA_instance_ready = RegionInstance::create_instance(
      arrayA_instance, dpu_mem, bounds, field_sizes, 1, ProfilingRequestSet());

  RegionInstance arrayB_instance = RegionInstance::NO_INST;
  Event arrayB_instance_ready = RegionInstance::create_instance(
      arrayB_instance, dpu_mem, bounds, field_sizes, 1, ProfilingRequestSet());

  RegionInstance arrayC_instance = RegionInstance::NO_INST;
  Event arrayC_instance_ready = RegionInstance::create_instance(
      arrayC_instance, dpu_mem, bounds, field_sizes, 1, ProfilingRequestSet());

  // ==== Data Movement ====
  Event arrayA_fill_done = Event::NO_EVENT;
  {
    std::vector<CopySrcDstField> srcs(1), dsts(1);

    // Fill the linear array with ones.
    srcs[0].set_fill<int>(1);
    dsts[0].set_field(arrayA_instance, 0, field_sizes[0]);
    arrayA_fill_done =
        bounds.copy(srcs, dsts, ProfilingRequestSet(), arrayA_instance_ready);

    // Overwrite the previous fill with the data from the array
    srcs[0].set_field(host_arrayA_instance, 0, field_sizes[0]);
    dsts[0].set_field(arrayA_instance, 0, field_sizes[0]);
    arrayA_fill_done =
        bounds.copy(srcs, dsts, ProfilingRequestSet(),
                    Event::merge_events(arrayA_fill_done, fill_task_done_event));
  }
  Event arrayB_fill_done = Event::NO_EVENT;
  {
    std::vector<CopySrcDstField> srcs(1), dsts(1);

    // Fill the linear array with ones.
    srcs[0].set_fill<int>(1);
    dsts[0].set_field(arrayB_instance, 0, field_sizes[0]);
    arrayB_fill_done =
        bounds.copy(srcs, dsts, ProfilingRequestSet(), arrayB_instance_ready);

    // Overwrite the previous fill with the data from the array
    srcs[0].set_field(host_arrayB_instance, 0, field_sizes[0]);
    dsts[0].set_field(arrayB_instance, 0, field_sizes[0]);
    arrayB_fill_done =
        bounds.copy(srcs, dsts, ProfilingRequestSet(),
                    Event::merge_events(arrayB_fill_done, fill_task_done_event));
  }

  Event arrayC_fill_done = Event::NO_EVENT;
  {
    std::vector<CopySrcDstField> srcs(1), dsts(1);

    // Fill the linear array with ones.
    srcs[0].set_fill<int>(1);
    dsts[0].set_field(arrayC_instance, 0, field_sizes[0]);
    arrayC_fill_done =
        bounds.copy(srcs, dsts, ProfilingRequestSet(), arrayC_instance_ready);
  }
  Event dpu_fill_done_event =
      Event::merge_events(arrayA_fill_done, arrayB_fill_done, arrayC_fill_done);

  // ==== Task Spawning ====
  Event cpu_task_done_event = Event::NO_EVENT;
  {
    DPU_TASK_ARGS args;
    args.bounds = bounds;
    args.arrayA_instance = host_arrayA_instance;
    args.arrayB_instance = host_arrayB_instance;
    args.arrayC_instance = host_arrayC_instance;
    cpu_task_done_event =
        cpu.spawn(CPU_LAUNCH_TASK, &args, sizeof(args), fill_task_done_event);
  };

  Event dpu_task_done_event = Event::NO_EVENT;
  {
    DPU_TASK_ARGS args;
    args.bounds = bounds;
    args.arrayA_instance = arrayA_instance;
    args.arrayB_instance = arrayB_instance;
    args.arrayC_instance = arrayC_instance;
    args.kernel = kern;
    dpu_task_done_event =
        dpu.spawn(DPU_LAUNCH_TASK, &args, sizeof(args), dpu_fill_done_event);
  };


  RegionInstance check_instance = RegionInstance::NO_INST;
  Event check_instance_ready_event = RegionInstance::create_instance(
      check_instance, cpu_mem, bounds, field_sizes, 1, ProfilingRequestSet());

  // Copy the result back, waiting on the processing to complete
  Event copy_done_event = Event::NO_EVENT;
  {
    std::vector<CopySrcDstField> srcs(1), dsts(1);
    // Initialize the host memory with some data
    srcs[0].set_fill<int>(1);
    dsts[0].set_field(check_instance, 0, field_sizes[0]);
    copy_done_event = bounds.copy(srcs, dsts, ProfilingRequestSet(),
                                  check_instance_ready_event);
    // Overwrite the previous fill with the data from the array
    srcs[0].set_field(arrayC_instance, 0, field_sizes[0]);
    dsts[0].set_field(check_instance, 0, field_sizes[0]);
    copy_done_event =
        bounds.copy(srcs, dsts, ProfilingRequestSet(),
                    Event::merge_events(copy_done_event, dpu_task_done_event));
  }


  Event check_task_done_event = Event::NO_EVENT;
  {
    CheckTaskArgs args;
    args.host_check_instance = host_arrayC_instance;
    args.device_check_instance = check_instance;
    check_task_done_event =
        cpu.spawn(CHECK_TASK, &args, sizeof(args), Event::merge_events(copy_done_event, cpu_task_done_event));
  }

  Runtime::get_runtime().shutdown(check_task_done_event);
}

int main(int argc, char **argv) {
  Runtime rt;
  
  rt.init(&argc, &argv);

  srand(time(NULL));

  Processor::register_task_by_kind(
      Processor::DPU_PROC, false /*!global*/, TOP_LEVEL_TASK,
      CodeDescriptor(top_level_task), ProfilingRequestSet(), 0, 0)
      .wait();

  Processor::register_task_by_kind(
      Processor::DPU_PROC, false /*!global*/, DPU_LAUNCH_TASK,
      CodeDescriptor(dpu_launch_task), ProfilingRequestSet(), 0, 0)
      .wait();

  Processor::register_task_by_kind(
      Processor::LOC_PROC, false /*!global*/, CPU_LAUNCH_TASK,
      CodeDescriptor(cpu_launch_task), ProfilingRequestSet(), 0, 0)
      .wait();


  Processor::register_task_by_kind(Processor::LOC_PROC, false /*!global*/,
                                   FILL_TASK, CodeDescriptor(fill_task),
                                   ProfilingRequestSet(), 0, 0)
      .wait();

  Processor::register_task_by_kind(Processor::LOC_PROC, false /*!global*/,
                                   CHECK_TASK, CodeDescriptor(check_task),
                                   ProfilingRequestSet(), 0, 0)
      .wait();

  Processor p = Machine::ProcessorQuery(Machine::get_machine())
                    .only_kind(Processor::DPU_PROC)
                    .first();

  assert((p != Processor::NO_PROC) && "Unable to find suitable DPU processor");

  rt.collective_spawn(p, TOP_LEVEL_TASK, 0, 0);

  return rt.wait_for_shutdown();
}