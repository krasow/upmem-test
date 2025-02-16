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

//TODO: the exp mod
//TODO: do we need the dynamic range handling?
//TODO: do we need warm_up iteration
//TODO: do we need exact the same program

#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <math.h>
#include <sys/time.h>

#include <legion.h>
#include <common.h>

#if !defined(LEGION_USE_UPMEM)
#error Legion not compiled with UPMEM enabled
#endif // !defined(LEGION_USE_UPMEM)

using namespace Legion;



#if defined(INT32)
#define RANDOM_NUMBER rand() % 16384
#define COMPARE(x, y) compare_int(x, y)
#define PRINT_EXPECTED(x, y) printf("expected %d, received %d --> ", x, y)

#elif defined(DOUBLE)
#define RANDOM_NUMBER drand48()
#define COMPARE(x, y) compare_double(x, y)
#define PRINT_EXPECTED(x, y) printf("expected %f, received %f --> ", x, y)
#endif

#define DEPTH 14
#define RANGE 16384
#define BINS 256

enum TaskIDs {
  TOP_LEVEL_TASK_ID,
  INIT_FIELD_TASK_ID,
  DAXPY_TASK_ID,
  CHECK_TASK_ID,
};

typedef struct {
  TYPE bins;
  TYPE depth;
  Realm::Upmem::Kernel *kernel;
} DPU_TASK_ARGS;

#define DPU_LAUNCH_BINARY "dpu/dpu_test_realm.up.o"

enum FieldIDs {
  FID_X,
  FID_Y,
};

typedef struct {
  TYPE x;
  TYPE y;
} daxpy_t;

double get_cur_time() {
  struct timeval tv;
  struct timezone tz;
  double cur_time;

  gettimeofday(&tv, &tz);
  cur_time = tv.tv_sec + tv.tv_usec / 1000000.0;

  return cur_time;
}

bool compare_double(double a, double b) {
  return fabs(a - b) < std::numeric_limits<double>::epsilon();
}

bool compare_int(int a, int b) { return a == b; }

void top_level_task(const Task *task,
                    const std::vector<PhysicalRegion> &regions, Context ctx,
                    Runtime *runtime) {
  Realm::Upmem::Kernel *kern = new Realm::Upmem::Kernel(DPU_LAUNCH_BINARY);
  kern->load();

  int num_elements = 16384;
  int num_subregions = 4;
  int soa_flag = 0;
  {
    const InputArgs &command_args = Runtime::get_input_args();
    for (int i = 1; i < command_args.argc; i++) {
      if (!strcmp(command_args.argv[i], "-n"))
        num_elements = atoi(command_args.argv[++i]);
      if (!strcmp(command_args.argv[i], "-b"))
        num_subregions = atoi(command_args.argv[++i]);
      if (!strcmp(command_args.argv[i], "-s"))
        soa_flag = atoi(command_args.argv[++i]);
    }
  }
  printf("Running HST-S for %d elements...\n", num_elements);
  printf("Partitioning data into %d sub-regions...\n", num_subregions);

  Rect<1> elem_rect(0, num_elements - 1);
  IndexSpace is = runtime->create_index_space(ctx, elem_rect);
  runtime->attach_name(is, "is");
  FieldSpace fs = runtime->create_field_space(ctx);
  runtime->attach_name(fs, "fs");
  {
    FieldAllocator allocator = runtime->create_field_allocator(ctx, fs);
    allocator.allocate_field(sizeof(TYPE), FID_X);
    runtime->attach_name(fs, FID_X, "X");
  }

  Rect<1> output_elem_rect(0, num_subregions*BINS - 1);
  IndexSpace output_is = runtime->create_index_space(ctx, output_elem_rect);
  runtime->attach_name(output_is, "output_is");
  FieldSpace output_fs = runtime->create_field_space(ctx);
  runtime->attach_name(output_fs, "output_fs");
  {
    FieldAllocator allocator = runtime->create_field_allocator(ctx, output_fs);
    allocator.allocate_field(sizeof(TYPE), FID_Y);
    runtime->attach_name(output_fs, FID_Y, "Y");
  }

  LogicalRegion input_lr = runtime->create_logical_region(ctx, is, fs);
  runtime->attach_name(input_lr, "input_lr");
  LogicalRegion output_lr = runtime->create_logical_region(ctx, output_is, output_fs);
  runtime->attach_name(output_lr, "output_lr");

  PhysicalRegion x_pr, y_pr;
  TYPE *x_ptr = NULL;
  TYPE *y_ptr = NULL;
  if (soa_flag == 0) { // SOA
    size_t x_bytes = sizeof(TYPE) * (num_elements);
    x_ptr = (TYPE *)malloc(x_bytes);
    size_t y_bytes = sizeof(TYPE) * (num_subregions*BINS);
    y_ptr = (TYPE *)malloc(y_bytes);
    for (int j = 0; j < num_elements; j++) {
      x_ptr[j] = 0;
    }
    for (int j = 0; j < num_subregions*BINS; j++) {
      y_ptr[j] = 0;
    }
    {
      printf("Attach SOA array fid %d, ptr %p\n", FID_X, x_ptr);
      AttachLauncher launcher(LEGION_EXTERNAL_INSTANCE, input_lr, input_lr);
      std::vector<FieldID> attach_fields(1);
      attach_fields[0] = FID_X;
      launcher.initialize_constraints(false /*column major*/, true /*soa*/,
                                      attach_fields);
      launcher.privilege_fields.insert(attach_fields.begin(),
                                       attach_fields.end());
      Realm::ExternalMemoryResource resource(x_ptr, x_bytes);
      launcher.external_resource = &resource;
      x_pr = runtime->attach_external_resource(ctx, launcher);
    }
    {
      printf("Attach SOA array fid %d, ptr %p\n", FID_Y, y_ptr);
      AttachLauncher launcher(LEGION_EXTERNAL_INSTANCE, output_lr, output_lr);
      std::vector<FieldID> attach_fields(1);
      attach_fields[0] = FID_Y;
      launcher.initialize_constraints(false /*column major*/, true /*soa*/,
                                      attach_fields);
      launcher.privilege_fields.insert(attach_fields.begin(),
                                       attach_fields.end());
      Realm::ExternalMemoryResource resource(y_ptr, y_bytes);
      launcher.external_resource = &resource;
      y_pr = runtime->attach_external_resource(ctx, launcher);
    }
  } 
  // else { // AOS
  //   size_t total_bytes = sizeof(daxpy_t) * (num_elements);
  //   daxpy_t *xyz_ptr = (daxpy_t *)malloc(total_bytes);
  //   std::vector<FieldID> layout_constraint_fields(3);
  //   layout_constraint_fields[0] = FID_X;
  //   layout_constraint_fields[1] = FID_Y;
  //   layout_constraint_fields[2] = FID_Z;
  //   // Need separate attaches for different logical regions,
  //   // each launcher gets all the fields in the layout constraint
  //   // but only requests privileges on fields for its logical region
  //   printf("Attach AOS array ptr %p\n", xyz_ptr);
  //   {
  //     AttachLauncher launcher(LEGION_EXTERNAL_INSTANCE, input_lr, input_lr);
  //     launcher.initialize_constraints(false /*column major*/, false /*soa*/,
  //                                     layout_constraint_fields);
  //     launcher.privilege_fields.insert(FID_X);
  //     launcher.privilege_fields.insert(FID_Y);
  //     Realm::ExternalMemoryResource resource(xyz_ptr, total_bytes);
  //     launcher.external_resource = &resource;
  //     xy_pr = runtime->attach_external_resource(ctx, launcher);
  //   }
  //   {
  //     AttachLauncher launcher(LEGION_EXTERNAL_INSTANCE, output_lr, output_lr);
  //     launcher.initialize_constraints(false /*columns major*/, false /*soa*/,
  //                                     layout_constraint_fields);
  //     launcher.privilege_fields.insert(FID_Z);
  //     Realm::ExternalMemoryResource resource(xyz_ptr, total_bytes);
  //     launcher.external_resource = &resource;
  //     z_pr = runtime->attach_external_resource(ctx, launcher);
  //   }
  // } 0
  

  Rect<1> color_bounds(0, num_subregions - 1);
  IndexSpace color_is = runtime->create_index_space(ctx, color_bounds);

  IndexPartition ip = runtime->create_equal_partition(ctx, is, color_is);
  runtime->attach_name(ip, "ip");
  IndexPartition output_ip = runtime->create_equal_partition(ctx, output_is, color_is);
  runtime->attach_name(output_ip, "output_ip");

  LogicalPartition input_lp = runtime->get_logical_partition(ctx, input_lr, ip);
  runtime->attach_name(input_lp, "input_lp");
  LogicalPartition output_lp = runtime->get_logical_partition(ctx, output_lr, output_ip);
  runtime->attach_name(output_lp, "output_lp");

  ArgumentMap arg_map;
  double start_init = get_cur_time();
  // printf("before the init task\n");

  IndexLauncher init_launcher(INIT_FIELD_TASK_ID, color_is,
                              TaskArgument(NULL, 0), arg_map);

  init_launcher.add_region_requirement(RegionRequirement(
      input_lp, 0 /*projection ID*/, WRITE_DISCARD, EXCLUSIVE, input_lr));
  init_launcher.region_requirements[0].add_field(FID_X);
  FutureMap fmi0 = runtime->execute_index_space(ctx, init_launcher);

  // init_launcher.region_requirements[0].privilege_fields.clear();
  // init_launcher.region_requirements[0].instance_fields.clear();
  // init_launcher.region_requirements[0].add_field(FID_Y);
  // FutureMap fmi1 = runtime->execute_index_space(ctx, init_launcher);
  // fmi1.wait_all_results();
  fmi0.wait_all_results();
  double end_init = get_cur_time();
  printf("Attach array, init done, time %f\n", end_init - start_init);

  // const TYPE alpha = RANDOM_NUMBER;
  double start_t = get_cur_time();

  DPU_TASK_ARGS args;
  args.bins = BINS;
  args.depth = DEPTH;
  args.kernel = kern;

  IndexLauncher daxpy_launcher(DAXPY_TASK_ID, color_is,
                               TaskArgument(&args, sizeof(DPU_TASK_ARGS)),
                               arg_map);
  daxpy_launcher.add_region_requirement(RegionRequirement(
      input_lp, 0 /*projection ID*/, READ_ONLY, EXCLUSIVE, input_lr));
  daxpy_launcher.region_requirements[0].add_field(FID_X);
  // daxpy_launcher.region_requirements[0].add_field(FID_Y);
  daxpy_launcher.add_region_requirement(RegionRequirement(
      output_lp, 0 /*projection ID*/, WRITE_DISCARD, EXCLUSIVE, output_lr));
  daxpy_launcher.region_requirements[1].add_field(FID_Y);
  FutureMap fm = runtime->execute_index_space(ctx, daxpy_launcher);
  fm.wait_all_results();
  double end_t = get_cur_time();
  printf("Attach array, HST done, time %f\n", end_t - start_t);


  TaskLauncher check_launcher(CHECK_TASK_ID,
                              TaskArgument(NULL, 0));
  check_launcher.add_region_requirement(
      RegionRequirement(input_lr, READ_ONLY, EXCLUSIVE, input_lr));
  check_launcher.region_requirements[0].add_field(FID_X);
  // check_launcher.region_requirements[0].add_field(FID_Y);
  check_launcher.add_region_requirement(
      RegionRequirement(output_lr, READ_ONLY, EXCLUSIVE, output_lr));
  check_launcher.region_requirements[1].add_field(FID_Y);
  Future fu = runtime->execute_task(ctx, check_launcher);
  fu.wait();

  runtime->detach_external_resource(ctx, x_pr);
  runtime->detach_external_resource(ctx, y_pr);
  runtime->destroy_logical_region(ctx, input_lr);
  runtime->destroy_logical_region(ctx, output_lr);
  runtime->destroy_field_space(ctx, fs);
  runtime->destroy_index_space(ctx, is);
  // if (xyz_ptr == NULL)
  //   free(xyz_ptr);
  if (x_ptr == NULL)
    free(x_ptr);
  if (y_ptr == NULL)
    free(y_ptr);
}

void init_field_task(const Task *task,
                     const std::vector<PhysicalRegion> &regions, Context ctx,
                     Runtime *runtime) {
  assert(regions.size() == 1);
  assert(task->regions.size() == 1);
  assert(task->regions[0].privilege_fields.size() == 1);

  FieldID fid = *(task->regions[0].privilege_fields.begin());
  const int point = task->index_point.point_data[0];
  printf("Initializing field %d for block %d...\n", fid, point);

  const AccessorWD acc(regions[0], fid);

  Rect<1> rect = runtime->get_index_space_domain(
      ctx, task->regions[0].region.get_index_space());
  for (PointInRectIterator<1> pir(rect); pir(); pir++)
    acc[*pir] = RANDOM_NUMBER;
}

void daxpy_task(const Task *task, const std::vector<PhysicalRegion> &regions,
                Context ctx, Runtime *runtime) {
  assert(regions.size() == 2);
  assert(task->regions.size() == 2);
  assert(task->arglen == sizeof(DPU_TASK_ARGS));
  DPU_TASK_ARGS task_args = *((DPU_TASK_ARGS *)task->args);
  const TYPE bins = task_args.bins;
  const TYPE depth = task_args.depth;
  const int point = task->index_point.point_data[0];

  const AccessorWD acc_y(regions[1], FID_Y);
  const AccessorRO acc_x(regions[0], FID_X);

  Rect<1> rect = runtime->get_index_space_domain(
      ctx, task->regions[0].region.get_index_space());
  Rect<1> rect_y = runtime->get_index_space_domain(
      ctx, task->regions[1].region.get_index_space());
  printf(
      "Running HST computation for point %d, xptr %p...\n",
      point, acc_x.ptr(rect.lo));
  printf(
      "The low value is %d\n",rect.lo.value);


  {
    DPU_LAUNCH_ARGS args;
    args.bins = bins;
    args.depth = depth;
    args.rect = rect;
    args.rect_y = rect_y;
    args.acc_y = acc_y;
    args.acc_x = acc_x;
    args.kernel = test;
    // launch specific upmem kernel
    task_args.kernel->launch((void **)&args, "ARGS", sizeof(DPU_LAUNCH_ARGS));
  }
}

void check_task(const Task *task, const std::vector<PhysicalRegion> &regions,
                Context ctx, Runtime *runtime) {
  assert(regions.size() == 2);
  assert(task->regions.size() == 2);
  // assert(task->arglen == sizeof(TYPE));
  // const TYPE alpha = *((const TYPE *)task->args);

  const AccessorRO acc_x(regions[0], FID_X);
  const AccessorRO acc_y(regions[1], FID_Y);
  // const AccessorRO acc_z(regions[1], FID_Z);

  Rect<1> rect_x = runtime->get_index_space_domain(
      ctx, task->regions[0].region.get_index_space());
  Rect<1> rect_y = runtime->get_index_space_domain(
      ctx, task->regions[1].region.get_index_space());



  int* sum_hst[BINS];
  for(int i=0; i<BINS; i++) sum_hst[i] = 0;
  for(PointInRectIterator<1> pir(rect_x); pir(); pir++){
    sum_hst[acc_x[*pir]*BINS>>DEPTH]++;
  }
#ifdef PRINT_UPMEM
    printf("the resulting values are:\n");
    for(PointInRectIterator<1> pir(rect_y); pir(); pir++){
      printf("%d\n", acc_y[*pir]);
    }
#endif
  fflush(stdout);

  int counter = 0;
  for(PointInRectIterator<1> pir(rect_y); pir(); pir++){
    sum_hst[counter%BINS]-=acc_y[*pir];
    counter++;
  }

  bool all_passed = true;
  for(int i=0; i<BINS; i++){
    if(sum_hst[i]!=0) all_passed = false;
  }
  
  if (all_passed)
    printf("SUCCESS!\n");
  else {
    printf("FAILURE!\n");
    abort();
  }
}

int main(int argc, char **argv) {
  Runtime::set_top_level_task_id(TOP_LEVEL_TASK_ID);

  {
    TaskVariantRegistrar registrar(TOP_LEVEL_TASK_ID, "top_level");
    registrar.add_constraint(ProcessorConstraint(Processor::DPU_PROC));
    Runtime::preregister_task_variant<top_level_task>(registrar, "top_level");
  }

  {
    TaskVariantRegistrar registrar(INIT_FIELD_TASK_ID, "init_field");
    registrar.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
    registrar.set_leaf();
    Runtime::preregister_task_variant<init_field_task>(registrar, "init_field");
  }

  {
    TaskVariantRegistrar registrar(DAXPY_TASK_ID, "daxpy");
    registrar.add_constraint(ProcessorConstraint(Processor::DPU_PROC));
    registrar.set_leaf();
    Runtime::preregister_task_variant<daxpy_task>(registrar, "daxpy");
  }

  {
    TaskVariantRegistrar registrar(CHECK_TASK_ID, "check");
    registrar.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
    registrar.set_leaf();
    Runtime::preregister_task_variant<check_task>(registrar, "check");
  }

  return Runtime::start(argc, argv);
}
