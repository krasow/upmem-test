#include "realm.h"
#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <thread>
#include <time.h>

using namespace Realm;

Logger log_app("app");

enum {
  TOP_LEVEL_TASK = Processor::TASK_ID_FIRST_AVAILABLE + 0,
  DPU_LAUNCH_TASK,
  CHECK_TASK
};

enum {
  DPU_LAUNCH_TASK_BINARY = "dpu_test.up.o"
}

struct DPUTaskArgs {
  int hi[128];
};

static void dpu_launch_task(const void *data, size_t datalen,
                            const void *userdata, size_t userlen, Processor p) {
  assert(datalen == sizeof(DPUTaskArgs));
  DPUTaskArgs task_args = *static_cast<const DPUTaskArgs *>(data);

  DPU_ASSERT(dpu_load( , DPU_LAUNCH_TASK_BINARY, NULL));
  

  
}

void top_level_task(const void *args, size_t arglen, const void *userdata,
                    size_t userlen, Processor p) {
  log_app.print() << "top task running on " << p;
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

  Processor::register_task_by_kind(Processor::LOC_PROC, false /*!global*/,
                                   CHECK_TASK, CodeDescriptor(check_task),
                                   ProfilingRequestSet(), 0, 0)
      .wait();

  Processor p = Machine::ProcessorQuery(Machine::get_machine())
                    .only_kind(Processor::DPU_PROC)
                    .first();

  assert((p != Processor::NO_PROC) && "Unable to find suitable DPU processor");

  Event e = rt.collective_spawn(p, TOP_LEVEL_TASK, 0, 0);
  rt.shutdown(e);
  return 0;

}
