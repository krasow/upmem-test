/* Copyright 2024 Stanford University, Los Alamos National Laboratory,
 *                Northwestern University
 * 
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <thread>
#include <time.h>
#include <unistd.h>

/* realm system library */
#include <realm.h> 
/* common header between device and host */
#include <common.h>

#if !defined(REALM_USE_UPMEM)
#error Realm not compiled with UPMEM enabled
#endif // !defined(REALM_USE_UPMEM)

using namespace Realm;

#define WIDTH 64
#define HEIGHT 64

Logger log_app("app");

// for realm
enum {
  TOP_LEVEL_TASK = Processor::TASK_ID_FIRST_AVAILABLE + 0,
  DPU_LAUNCH_TASK,
  CPU_LAUNCH_TASK,
  FILL_TASK,
  CHECK_TASK
};

enum FieldIDs {
  FID_A = 101,
  FID_B = 102,
  FID_C = 103,
};

typedef struct {
  Rect<2> bounds;
  RegionInstance linear_instance;
  Upmem::Kernel* kernel;
} __attribute__((aligned(8))) DPU_TASK_ARGS;

// for the device
#define DPU_LAUNCH_BINARY "dpu/dpu_test_realm.up.o"


static void dpu_launch_task(const void *data, size_t datalen,
                            const void *userdata, size_t userlen,
                            Processor dpu) {

  log_app.print() << "dpu launch task running on " << dpu;

  assert(datalen == sizeof(DPU_TASK_ARGS));
  DPU_TASK_ARGS task_args = *static_cast<const DPU_TASK_ARGS *>(data);

  AffineAccessor<TYPE, 2> arrayA_accessor(task_args.linear_instance, FID_A);
  AffineAccessor<TYPE, 2> arrayB_accessor(task_args.linear_instance, FID_B);
  AffineAccessor<TYPE, 2> arrayC_accessor(task_args.linear_instance, FID_C);

  {
    DPU_LAUNCH_ARGS args;
    args.bounds = task_args.bounds;
    args.arrayA_accessor = arrayA_accessor;
    args.arrayB_accessor = arrayB_accessor;
    args.arrayC_accessor = arrayC_accessor;
    args.kernel = test;
    // launch specific upmem kernel
    task_args.kernel->launch((void **)&args, "ARGS", sizeof(DPU_LAUNCH_ARGS));
  }
}

static void cpu_launch_task(const void *data, size_t datalen,
                            const void *userdata, size_t userlen,
                            Processor cpu) {

  log_app.print() << "cpu launch task running on " << cpu;

  assert(datalen == sizeof(DPU_TASK_ARGS));
  DPU_TASK_ARGS task_args = *static_cast<const DPU_TASK_ARGS *>(data);

  AffineAccessor<TYPE, 2> arrayA_accessor(task_args.linear_instance, FID_A);
  AffineAccessor<TYPE, 2> arrayB_accessor(task_args.linear_instance, FID_B);
  AffineAccessor<TYPE, 2> arrayC_accessor(task_args.linear_instance, FID_C);

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
  RegionInstance linear_instance;
};

static void fill_task(const void *data, size_t datalen, const void *userdata,
                       size_t userlen, Processor p) {
  assert(datalen == sizeof(FillTaskArgs));
  const FillTaskArgs &task_args = *static_cast<const FillTaskArgs *>(data);

  AffineAccessor<TYPE, 2> arrayA_accessor(task_args.linear_instance, FID_A);
  AffineAccessor<TYPE, 2> arrayB_accessor(task_args.linear_instance, FID_B);

  const Rect<2> bounds =
      task_args.linear_instance.get_indexspace<2>().bounds;

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
  AffineAccessor<TYPE, 2> host_linear_accessor(task_args.host_check_instance, FID_C);
  AffineAccessor<TYPE, 2> device_linear_accessor(task_args.device_check_instance, 0);
  
  // bounds are the same in both cases
  const Rect<2> bounds =
        task_args.device_check_instance.get_indexspace<2>().bounds;

  {
    FILE *file = fopen("array.bin", "wb");
    assert(file != NULL && "error open file");
    for (PointInRectIterator<2> pir(bounds); pir.valid; pir.step()) {
      TYPE value = device_linear_accessor[pir.p];
      fwrite(&value, sizeof(TYPE), 1, file);
    }
    fclose(file);
  };
  {
    FILE *file = fopen("host_array.bin", "wb");
    assert(file != NULL && "error open file");
    for (PointInRectIterator<2> pir(bounds); pir.valid; pir.step()) {
      TYPE value = host_linear_accessor[pir.p];
      fwrite(&value, sizeof(TYPE), 1, file);
    }
    fclose(file);
  };

  for (PointInRectIterator<2> pir(bounds); pir.valid; pir.step()) {
    TYPE correct = host_linear_accessor[pir.p];
    TYPE device_check = device_linear_accessor[pir.p];
    assert(correct == device_check && "ERROR: device does not equal host");
  }

}

void top_level_task(const void *args, size_t arglen, const void *userdata,
                    size_t userlen, Processor dpu) {
  log_app.print() << "top task running on " << dpu;

  Upmem::Kernel* kern = new Upmem::Kernel(DPU_LAUNCH_BINARY);
  // the binary needs to be loaded before any memory operations
  kern->load();

  const size_t width = WIDTH, height = HEIGHT;

  std::map<FieldID, size_t> field_sizes;
  field_sizes[FID_A] = sizeof(TYPE);
  field_sizes[FID_B] = sizeof(TYPE);
  field_sizes[FID_C] = sizeof(TYPE);

  Rect<2> bounds(Point<2>(0, 0), Point<2>(width - 1, height - 1));


  Processor cpu = Machine::ProcessorQuery(Machine::get_machine())
                      .only_kind(Processor::LOC_PROC)
                      .local_address_space()
                      .first();
  assert((cpu != Processor::NO_PROC) &&
         "Failed to find suitable CPU processor to check results!");

  Memory cpu_mem = Machine::MemoryQuery(Machine::get_machine())
                       .has_capacity(4 * bounds.volume() * sizeof(TYPE))
                       .has_affinity_to(cpu)
                       .has_affinity_to(dpu)
                       .first();

  assert((cpu_mem != Memory::NO_MEMORY) &&
         "Failed to find suitable CPU memory to use!");

  std::cout << "Choosing CPU memory type " << cpu_mem.kind()
            << " for CPU processor " << cpu << std::endl;

  RegionInstance host_linear_instance = RegionInstance::NO_INST;
  Event host_linear_instance_ready =
      RegionInstance::create_instance(host_linear_instance, cpu_mem, bounds,
                                      field_sizes, 0, ProfilingRequestSet());
  
  // ==== Data Movement ====
  Event host_fill_done_event = Event::NO_EVENT;
  {
    std::vector<CopySrcDstField> srcs(3), dsts(3);

    srcs[0].set_fill<TYPE>(1);
    dsts[0].set_field(host_linear_instance, FID_A, field_sizes[FID_A]);

    srcs[1].set_fill<TYPE>(1);
    dsts[1].set_field(host_linear_instance, FID_B, field_sizes[FID_B]);

    srcs[2].set_fill<TYPE>(1);
    dsts[2].set_field(host_linear_instance, FID_C, field_sizes[FID_C]);

    host_fill_done_event = bounds.copy(srcs, dsts, ProfilingRequestSet(),
                                        host_linear_instance_ready);
  }


  Event fill_task_done_event = Event::NO_EVENT;
  {
    FillTaskArgs args;
    args.linear_instance = host_linear_instance;
    fill_task_done_event =
        cpu.spawn(FILL_TASK, &args, sizeof(args), host_fill_done_event);
  }

  // ==== Allocating DPU memory with Realm ====
  Memory dpu_mem = Machine::MemoryQuery(Machine::get_machine())
                       .has_capacity(bounds.volume() /* index space */ *
                                     3 /* number of fields */ *
                                     sizeof(TYPE) /* type of field */)
                       .best_affinity_to(dpu)
                       .first();

  assert((dpu_mem != Memory::NO_MEMORY) &&
         "Failed to find suitable DPU memory to use for bound 3!");

  // Now create a 2D instance like we normally would
  RegionInstance device_linear_instance = RegionInstance::NO_INST;
  Event device_instance_ready = RegionInstance::create_instance(
      device_linear_instance, dpu_mem, bounds, field_sizes, 0, ProfilingRequestSet());

  // ==== Data Movement ====
  Event device_init_field_event = Event::NO_EVENT;
  {
    std::vector<CopySrcDstField> srcs(3), dsts(3);

    // Fill the linear array with ones.
    srcs[0].set_fill<TYPE>(1);
    dsts[0].set_field(device_linear_instance, FID_A, field_sizes[FID_A]);

    srcs[1].set_fill<TYPE>(1);
    dsts[1].set_field(device_linear_instance, FID_B, field_sizes[FID_B]);

    srcs[2].set_fill<TYPE>(1);
    dsts[2].set_field(device_linear_instance, FID_C, field_sizes[FID_C]);
    device_init_field_event =
        bounds.copy(srcs, dsts, ProfilingRequestSet(), device_instance_ready);
  }

  Event device_copy_field_event = Event::NO_EVENT;
  {
    std::vector<CopySrcDstField> srcs(2), dsts(2);

    // Overwrite the previous fill with the data from the array
    srcs[0].set_field(host_linear_instance, FID_A, field_sizes[FID_A]);
    dsts[0].set_field(device_linear_instance, FID_A, field_sizes[FID_A]);

    srcs[1].set_field(host_linear_instance, FID_B, field_sizes[FID_B]);
    dsts[1].set_field(device_linear_instance, FID_B, field_sizes[FID_B]);
    
    device_copy_field_event =
        bounds.copy(srcs, dsts, ProfilingRequestSet(),
                    Event::merge_events(device_init_field_event, fill_task_done_event));
  }


  // ==== Task Spawning ====
  Event cpu_task_done_event = Event::NO_EVENT;
  {
    DPU_TASK_ARGS args;
    args.bounds = bounds;
    args.linear_instance = host_linear_instance;
    cpu_task_done_event =
        cpu.spawn(CPU_LAUNCH_TASK, &args, sizeof(args), fill_task_done_event);
  };

  Event dpu_task_done_event = Event::NO_EVENT;
  {
    DPU_TASK_ARGS args;
    args.bounds = bounds;
    args.linear_instance = device_linear_instance;
    args.kernel = kern;
    dpu_task_done_event =
        dpu.spawn(DPU_LAUNCH_TASK, &args, sizeof(args), device_copy_field_event);
  };

  std::vector<size_t> check_field_size(1, sizeof(TYPE));
  RegionInstance check_instance = RegionInstance::NO_INST;
  Event check_instance_ready_event = RegionInstance::create_instance(
      check_instance, cpu_mem, bounds, check_field_size, 0, ProfilingRequestSet());

  // Copy the result back, waiting on the processing to complete
  Event copy_done_event = Event::NO_EVENT;
  {
    std::vector<CopySrcDstField> srcs(1), dsts(1);
    // Initialize the host memory with some data
    srcs[0].set_fill<TYPE>(1);
    dsts[0].set_field(check_instance, 0, check_field_size[0]);
    copy_done_event = bounds.copy(srcs, dsts, ProfilingRequestSet(),
                                  check_instance_ready_event);
    // Overwrite the previous fill with the data from the array
    srcs[0].set_field(device_linear_instance, FID_C, field_sizes[FID_C]);
    dsts[0].set_field(check_instance, 0, check_field_size[0]);
    copy_done_event =
        bounds.copy(srcs, dsts, ProfilingRequestSet(),
                    Event::merge_events(copy_done_event, dpu_task_done_event));
  }


  Event check_task_done_event = Event::NO_EVENT;
  {
    CheckTaskArgs args;
    args.host_check_instance = host_linear_instance;
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