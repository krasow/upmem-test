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
#include <limits>
#include <math.h>
#include <sys/time.h>

/* legion programming system library */
#include <legion.h>
/* common header between device and host */
#include <common.h>

#if !defined(LEGION_USE_UPMEM)
#error Legion not compiled with UPMEM enabled
#endif // !defined(LEGION_USE_UPMEM)

using namespace Legion;

#if defined(INT32)
#define RANDOM_NUMBER rand() % 8192
#define COMPARE(x, y) compare_int(x, y)
#define PRINT_EXPECTED(x, y) printf("expected %d, received %d --> ", x, y)

#elif defined(DOUBLE)
#define RANDOM_NUMBER drand48()
#define COMPARE(x, y) compare_double(x, y)
#define PRINT_EXPECTED(x, y) printf("expected %f, received %f --> ", x, y)
#endif

enum TaskIDs {
  TOP_LEVEL_TASK_ID,
  INIT_FIELD_TASK_ID,
  INIT_FIELD_TRANSPOSE_TASK_ID,
  MULTIPLY_TASK_ID,
  CHECK_TASK_ID,
};

typedef struct {
  TYPE alpha;
  Realm::Upmem::Kernel *kernel;
} DPU_TASK_ARGS;

// for the device
#define DPU_LAUNCH_BINARY "dpu/dpu_test_realm.up.o"

enum FieldIDs {
  FID_X,
  FID_Y,
  FID_Z,
};

typedef struct {
  TYPE x;
  TYPE y;
  TYPE z;
} multiply_t;

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

// #define NUM_SUBREGIONS 2

void print_mat(TYPE *ptr, int num)
{
  // printf("printing a matrix x\n");
  for(int i=0; i<num; i++){
    if(i%WIDTH == 0) printf("\n");
    printf("%f ", ptr[i]);
  }
  printf("\n");
}

void top_level_task(const Task *task,
                    const std::vector<PhysicalRegion> &regions, Context ctx,
                    Runtime *runtime) {
  Realm::Upmem::Kernel *kern = new Realm::Upmem::Kernel(DPU_LAUNCH_BINARY);
  // the binary needs to be loaded before any memory operations
  kern->load();

  int num_elements = WIDTH*HEIGHT;
  int num_subregions = NUM_SUBREGIONS;
  int soa_flag = 0;

  // See if we have any command line arguments to parse
  // Note we now have a new command line parameter which specifies
  // how many subregions we should make.
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
  printf("Running mat multiplication for %d elements...\n", num_elements);
  printf("Partitioning data into %d sub-regions...\n", num_subregions);
  
  //Duplicating the elements
  int num_elements_xy = num_elements*2;

  //creating two separate index spaces for xy and z
  Rect<1> elem_rect_xy(0, num_elements_xy - 1);
  IndexSpace is_xy = runtime->create_index_space(ctx, elem_rect_xy);
  runtime->attach_name(is_xy, "is_xy");
  Rect<1> elem_rect_z(0, num_elements - 1);
  IndexSpace is_z = runtime->create_index_space(ctx, elem_rect_z); 


  FieldSpace fs = runtime->create_field_space(ctx);
  runtime->attach_name(fs, "fs");
  {
    FieldAllocator allocator = runtime->create_field_allocator(ctx, fs);
    allocator.allocate_field(sizeof(TYPE), FID_X);
    runtime->attach_name(fs, FID_X, "X");
    allocator.allocate_field(sizeof(TYPE), FID_Y);
    runtime->attach_name(fs, FID_Y, "Y");
    allocator.allocate_field(sizeof(TYPE), FID_Z);
    runtime->attach_name(fs, FID_Z, "Z");
  }
  LogicalRegion input_lr = runtime->create_logical_region(ctx, is_xy, fs);
  runtime->attach_name(input_lr, "input_lr");
  LogicalRegion output_lr = runtime->create_logical_region(ctx, is_z, fs);
  runtime->attach_name(output_lr, "output_lr");

  PhysicalRegion xy_pr, z_pr;
  TYPE *z_ptr = NULL;
  TYPE *xy_ptr = NULL;
  TYPE *xyz_ptr = NULL;
  if (soa_flag == 0) { // SOA
    size_t xy_bytes = 2 /*fields*/ * sizeof(TYPE) * (num_elements_xy);
    xy_ptr = (TYPE *)malloc(xy_bytes);
    size_t z_bytes = sizeof(TYPE) * (num_elements);
    z_ptr = (TYPE *)malloc(z_bytes);
    for (int j = 0; j < num_elements_xy; j++) {
      xy_ptr[j] = RANDOM_NUMBER;
      xy_ptr[num_elements + j] = RANDOM_NUMBER;
    }

    for (int j = 0; j < num_elements; j++) {
      z_ptr[j] = RANDOM_NUMBER;
    }
    {
      printf("Attach SOA array fid %d, fid %d, ptr %p\n", FID_X, FID_Y, xy_ptr);
      AttachLauncher launcher(LEGION_EXTERNAL_INSTANCE, input_lr, input_lr);
      std::vector<FieldID> attach_fields(2);
      attach_fields[0] = FID_X;
      attach_fields[1] = FID_Y;
      launcher.initialize_constraints(false /*column major*/, true /*soa*/,
                                      attach_fields);
      launcher.privilege_fields.insert(attach_fields.begin(),
                                       attach_fields.end());
      Realm::ExternalMemoryResource resource(xy_ptr, xy_bytes);
      launcher.external_resource = &resource;
      xy_pr = runtime->attach_external_resource(ctx, launcher);
    }
    {
      printf("Attach SOA array fid %d, ptr %p\n", FID_Z, z_ptr);
      AttachLauncher launcher(LEGION_EXTERNAL_INSTANCE, output_lr, output_lr);
      std::vector<FieldID> attach_fields(1);
      attach_fields[0] = FID_Z;
      launcher.initialize_constraints(false /*column major*/, true /*soa*/,
                                      attach_fields);
      launcher.privilege_fields.insert(attach_fields.begin(),
                                       attach_fields.end());
      Realm::ExternalMemoryResource resource(z_ptr, z_bytes);
      launcher.external_resource = &resource;
      z_pr = runtime->attach_external_resource(ctx, launcher);
    }
  } else { // AOS
    size_t total_bytes = sizeof(multiply_t) * (num_elements);
    multiply_t *xyz_ptr = (multiply_t *)malloc(total_bytes);
    std::vector<FieldID> layout_constraint_fields(3);
    layout_constraint_fields[0] = FID_X;
    layout_constraint_fields[1] = FID_Y;
    layout_constraint_fields[2] = FID_Z;
    // Need separate attaches for different logical regions,
    // each launcher gets all the fields in the layout constraint
    // but only requests privileges on fields for its logical region
    printf("Attach AOS array ptr %p\n", xyz_ptr);
    {
      AttachLauncher launcher(LEGION_EXTERNAL_INSTANCE, input_lr, input_lr);
      launcher.initialize_constraints(false /*column major*/, false /*soa*/,
                                      layout_constraint_fields);
      launcher.privilege_fields.insert(FID_X);
      launcher.privilege_fields.insert(FID_Y);
      Realm::ExternalMemoryResource resource(xyz_ptr, total_bytes);
      launcher.external_resource = &resource;
      xy_pr = runtime->attach_external_resource(ctx, launcher);
    }
    {
      AttachLauncher launcher(LEGION_EXTERNAL_INSTANCE, output_lr, output_lr);
      launcher.initialize_constraints(false /*columns major*/, false /*soa*/,
                                      layout_constraint_fields);
      launcher.privilege_fields.insert(FID_Z);
      Realm::ExternalMemoryResource resource(xyz_ptr, total_bytes);
      launcher.external_resource = &resource;
      z_pr = runtime->attach_external_resource(ctx, launcher);
    }
  }




  // PhysicalRegion x_pr, y_pr, z_pr;
  // TYPE *z_ptr = NULL;
  // TYPE *x_ptr = NULL;
  // TYPE *y_ptr = NULL;
  // TYPE *xyz_ptr = NULL;
  // if (soa_flag == 0) { // SOA
  //   size_t xy_bytes = sizeof(TYPE) * (num_elements * num_elements);
  //   size_t z_bytes = sizeof(TYPE) * (num_elements);
  //   x_ptr = (TYPE *)malloc(xy_bytes);
  //   y_ptr = (TYPE *)malloc(xy_bytes);
  //   z_ptr = (TYPE *)malloc(z_bytes);
  //   for (int j = 0; j < num_elements; j++) {
  //     z_ptr[j] = 1;
  //   }
  //   for (int j = 0; j < num_elements_xy; j++) {
  //     x_ptr[j] = 1;
  //     y_ptr[j] = 1;
  //   }
       
  //   {
  //     printf("Attach SOA array fid %d, ptr %p\n", FID_X, x_ptr);
  //     AttachLauncher launcher(LEGION_EXTERNAL_INSTANCE, input_lr_X, input_lr_X);
  //     std::vector<FieldID> attach_fields(1);
  //     attach_fields[0] = FID_X;
  //     launcher.initialize_constraints(false /*column major*/, true /*soa*/,
  //                                     attach_fields);
  //     launcher.privilege_fields.insert(attach_fields.begin(),
  //                                      attach_fields.end());
  //     Realm::ExternalMemoryResource resource(x_ptr, xy_bytes);
  //     launcher.external_resource = &resource;
  //     x_pr = runtime->attach_external_resource(ctx, launcher);
  //   }
  //   {
  //     printf("Attach SOA array fid %d, ptr %p\n", FID_Y, y_ptr);
  //     AttachLauncher launcher(LEGION_EXTERNAL_INSTANCE, input_lr_Y, input_lr_Y);
  //     std::vector<FieldID> attach_fields(1);
  //     attach_fields[0] = FID_Y;
  //     launcher.initialize_constraints(false /*column major*/, true /*soa*/,
  //                                     attach_fields);
  //     launcher.privilege_fields.insert(attach_fields.begin(),
  //                                      attach_fields.end());
  //     Realm::ExternalMemoryResource resource(y_ptr, xy_bytes);
  //     launcher.external_resource = &resource;
  //     y_pr = runtime->attach_external_resource(ctx, launcher);
  //   }
  //   {
  //     printf("Attach SOA array fid %d, ptr %p\n", FID_Z, z_ptr);
  //     AttachLauncher launcher(LEGION_EXTERNAL_INSTANCE, output_lr, output_lr);
  //     std::vector<FieldID> attach_fields(1);
  //     attach_fields[0] = FID_Z;
  //     launcher.initialize_constraints(false /*column major*/, true /*soa*/,
  //                                     attach_fields);
  //     launcher.privilege_fields.insert(attach_fields.begin(),
  //                                      attach_fields.end());
  //     Realm::ExternalMemoryResource resource(z_ptr, z_bytes);
  //     launcher.external_resource = &resource;
  //     z_pr = runtime->attach_external_resource(ctx, launcher);
  //   }
  // } 

  // In addition to using rectangles and domains for launching index spaces
  // of tasks (see example 02), Legion also uses them for performing
  // operations on logical regions.  Here we create a rectangle and a
  // corresponding domain for describing the space of subregions that we
  // want to create.  Each subregion is assigned a 'color' which is why
  // we name the variables 'color_bounds' and 'color_domain'.  We'll use
  // these below when we partition the region.
  Rect<1> color_bounds(0, num_subregions*num_subregions - 1);
  IndexSpace color_is = runtime->create_index_space(ctx, color_bounds);

  // Parallelism in Legion is implicit.  This means that rather than
  // explicitly saying what should run in parallel, Legion applications
  // partition up data and tasks specify which regions they access.
  // The Legion runtime computes non-interference as a function of
  // regions, fields, and privileges and then determines which tasks
  // are safe to run in parallel.
  //
  // Data partitioning is performed on index spaces.  The partitioning
  // operation is used to break an index space of points into subsets
  // of points each of which will become a sub index space.  Partitions
  // created on an index space are then transitively applied to all the
  // logical regions created using the index space.  We will show how
  // to get names to the subregions later in this example.
  //
  // Here we want to create the IndexPartition 'ip'.  We'll illustrate
  // two ways of creating an index partition depending on whether the
  // array being partitioned can be evenly partitioned into subsets
  // or not.  There are other methods to partitioning index spaces
  // which are not covered here.  We'll cover the case of coloring
  // individual points in an index space in our capstone circuit example.
  IndexPartition ip = runtime->create_equal_partition(ctx, is_xy, color_is);
  runtime->attach_name(ip, "ip");

  // The index space 'is' was used in creating two logical regions: 'input_lr'
  // and 'output_lr'.  By creating an IndexPartitiong of 'is' we implicitly
  // created a LogicalPartition for each of the logical regions created using
  // 'is'.  The Legion runtime provides several ways of getting the names for
  // these LogicalPartitions.  We'll look at one of them here.  The
  // 'get_logical_partition' method takes a LogicalRegion and an IndexPartition
  // and returns the LogicalPartition of the given LogicalRegion that
  // corresponds to the given IndexPartition.
  LogicalPartition input_lp  = runtime->get_logical_partition(ctx, input_lr, ip);
  runtime->attach_name(input_lp , "input_lr");
  // LogicalPartition input_lp_Y_init = runtime->get_logical_partition(ctx, input_lr_Y, ip);
  // runtime->attach_name(input_lp_Y_init, "input_lp_Y_init");


  ArgumentMap arg_map_init;
  for(int i=0; i<num_subregions*num_subregions; i++){
    arg_map_init.set_point(Point<1>(i), TaskArgument(&i, sizeof(int)));
  }


  double start_init = get_cur_time();

  printf("the starting point of the init task\n");
  IndexLauncher init_launcher(INIT_FIELD_TASK_ID, color_is,
                              TaskArgument(NULL, 0), arg_map_init);

  init_launcher.add_region_requirement(RegionRequirement(
      input_lp, 0 /*projection ID*/, WRITE_DISCARD, EXCLUSIVE, input_lr));
  init_launcher.region_requirements[0].add_field(FID_X);
  FutureMap fm_X_init = runtime->execute_index_space(ctx, init_launcher);

  init_launcher.region_requirements[0].privilege_fields.clear();
  init_launcher.region_requirements[0].instance_fields.clear();
  init_launcher.region_requirements[0].add_field(FID_Y);
  FutureMap fm_Y_init = runtime->execute_index_space(ctx, init_launcher);
  fm_Y_init.wait_all_results();

  ArgumentMap t_argmap;
  Rect<1> transpose_color(0, 1);
  IndexSpace transpose_color_is = runtime->create_index_space(ctx, transpose_color);


  IndexLauncher transpose(INIT_FIELD_TRANSPOSE_TASK_ID, transpose_color_is,
                              TaskArgument(NULL, 0), t_argmap);

  transpose.add_region_requirement(RegionRequirement(
      input_lp, 0 /*projection ID*/, WRITE_DISCARD, EXCLUSIVE, input_lr));
  transpose.region_requirements[0].add_field(FID_Y);
  FutureMap fmi_Y_transpose = runtime->execute_index_space(ctx, transpose);
  fmi_Y_transpose.wait_all_results();
  fm_X_init.wait_all_results();


  double end_init = get_cur_time();
  printf("Attach array, init done, time %f\n", end_init - start_init);

  IndexPartition ip_z = runtime->create_equal_partition(ctx, is_z, color_is);
  runtime->attach_name(ip_z, "ip_z");
  
  LogicalPartition output_lp = runtime->get_logical_partition(ctx, output_lr, ip_z);
  runtime->attach_name(output_lp, "output_lp");

  // Create our launch domain.  Note that is the same as color domain
  // as we are going to launch one task for each subregion we created.
  ArgumentMap arg_map;

  // As in previous examples, we now want to launch tasks for initializing
  // both the fields.  However, to increase the amount of parallelism
  // exposed to the runtime we will launch separate sub-tasks for each of
  // the logical subregions created by our partitioning.  To express this
  // we create an IndexLauncher for launching an index space of tasks
  // the same as example 02.
  

  const TYPE alpha = RANDOM_NUMBER;
  double start_t = get_cur_time();

  DPU_TASK_ARGS args;
  args.alpha = alpha;
  args.kernel = kern;
  // We launch the subtasks for performing the multiply computation
  // in a similar way to the initialize field tasks.  Note we
  // again make use of two RegionRequirements which use a
  // partition as the upper bound for the privileges for the task.
  IndexLauncher matrix_multi_launcher(MULTIPLY_TASK_ID, color_is,
                               TaskArgument(&args, sizeof(DPU_TASK_ARGS)),
                               arg_map);
  matrix_multi_launcher.add_region_requirement(RegionRequirement(
      input_lp, 0 /*projection ID*/, READ_ONLY, EXCLUSIVE, input_lr));
  matrix_multi_launcher.region_requirements[0].add_field(FID_X);
  matrix_multi_launcher.region_requirements[0].add_field(FID_Y);
  matrix_multi_launcher.add_region_requirement(RegionRequirement(
      output_lp, 0 /*projection ID*/, WRITE_DISCARD, EXCLUSIVE, output_lr));
  matrix_multi_launcher.region_requirements[1].add_field(FID_Z);
  FutureMap fm = runtime->execute_index_space(ctx, matrix_multi_launcher);
  fm.wait_all_results();
  double end_t = get_cur_time();
  printf("Attach array, mat multiplication done, time %f\n", end_t - start_t);

  // While we could also issue parallel subtasks for the checking
  // task, we only issue a single task launch to illustrate an
  // important Legion concept.  Note the checking task operates
  // on the entire 'input_lr' and 'output_lr' regions and not
  // on the subregions.  Even though the previous tasks were
  // all operating on subregions, Legion will correctly compute
  // data dependences on all the subtasks that generated the
  // data in these two regions.
  TaskLauncher check_launcher(CHECK_TASK_ID,
                              TaskArgument(&alpha, sizeof(alpha)));
  check_launcher.add_region_requirement(RegionRequirement(input_lr, READ_ONLY, EXCLUSIVE, input_lr));
  check_launcher.region_requirements[0].add_field(FID_X);
  check_launcher.region_requirements[0].add_field(FID_Y);
  check_launcher.add_region_requirement(RegionRequirement(output_lr, READ_ONLY, EXCLUSIVE, output_lr));
  check_launcher.region_requirements[1].add_field(FID_Z);
  Future fu = runtime->execute_task(ctx, check_launcher);
  fu.wait();

  runtime->detach_external_resource(ctx, xy_pr);
  runtime->detach_external_resource(ctx, z_pr);
  runtime->destroy_logical_region(ctx, input_lr);
  runtime->destroy_logical_region(ctx, output_lr);
  runtime->destroy_field_space(ctx, fs);
  runtime->destroy_index_space(ctx, is_xy);
  runtime->destroy_index_space(ctx, is_z);

  if (xy_ptr == NULL)
    free(xy_ptr);
  if (z_ptr == NULL)
    free(z_ptr);
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
  int subregion_id = *((const int *)task->local_args);
  printf("the argument value that I got is %d \n", subregion_id);

  const AccessorWD acc(regions[0], fid);

  // Note here that we get the domain for the subregion for
  // this task from the runtime which makes it safe for running
  // both as a single task and as part of an index space of tasks.
  Rect<1> rect = runtime->get_index_space_domain(
      ctx, task->regions[0].region.get_index_space());
  
  TYPE val = subregion_id + 1.0;
  if (subregion_id % 2 != 0) { val = (TYPE)subregion_id + 0.0; }

  for (PointInRectIterator<1> pir(rect); pir(); pir++)
    acc[*pir] = val;

}

void init_field_task_transpose(const Task *task,
                     const std::vector<PhysicalRegion> &regions, Context ctx,
                     Runtime *runtime) {
  assert(regions.size() == 1);
  assert(task->regions.size() == 1);
  assert(task->regions[0].privilege_fields.size() == 1);

  FieldID fid = *(task->regions[0].privilege_fields.begin());
  const int point = task->index_point.point_data[0];
  printf("Initializing transpose field %d for block %d...\n", fid, point);

  const AccessorWD acc(regions[0], fid);

  // Note here that we get the domain for the subregion for
  // this task from the runtime which makes it safe for running
  // both as a single task and as part of an index space of tasks.
  Rect<1> rect = runtime->get_index_space_domain(
      ctx, task->regions[0].region.get_index_space());

  // if(rect.lo == Point<1>(0)){
    // for (PointInRectIterator<1> pir(rect); pir(); pir++)
    //   acc[*pir] = RANDOM_NUMBER;
  // }else{
    // int size_duplication = HEIGHT*WIDTH;
    // Rect<1> rect_ori;
    // //? there would be multiple threads reading the same data, will it cause some problem
    // rect_ori.lo = Point<1>(0);
    // rect_ori.hi = Point<1>(HEIGHT*WIDTH - 1);
    // PointInRectIterator<1> pir_ori(rect_ori);
    // for (PointInRectIterator<1> pir(rect); pir(); pir++){
    //   acc[*pir] = acc[*pir_ori];
    //   pir_ori++;
    // }

  // }
}

void multiply_task(const Task *task, const std::vector<PhysicalRegion> &regions,
                Context ctx, Runtime *runtime) {
  assert(regions.size() == 2);
  assert(task->regions.size() == 2);
  assert(task->arglen == sizeof(DPU_TASK_ARGS));
  DPU_TASK_ARGS task_args = *((DPU_TASK_ARGS *)task->args);
  const TYPE alpha = task_args.alpha;
  const int point = task->index_point.point_data[0];
  const AccessorRO acc_x(regions[0], FID_X);
  const AccessorRO acc_y(regions[0], FID_Y);
  const AccessorWD acc_z(regions[1], FID_Z);

  Rect<1> rect = runtime->get_index_space_domain(
      ctx, task->regions[0].region.get_index_space());
  printf(
      "Running mat multipilication for point %d, xptr %p, y_ptr %p, z_ptr %p...",
      point, acc_x.ptr(rect.lo), acc_y.ptr(rect.lo), acc_z.ptr(rect.lo));
#ifdef INT32
  printf(" alpha = %d \n", alpha);
#elif DOUBLE
  printf(" alpha = %f \n", alpha);
#endif

  {
    DPU_LAUNCH_ARGS args;
    // args.width = WIDTH;
    // args.height = HEIGHT;
    args.alpha = alpha;
    args.rect = rect;
    args.acc_y = acc_y;
    args.acc_x = acc_x;
    args.acc_z = acc_z;
    args.kernel = test;
    // launch specific upmem kernel
    task_args.kernel->launch((void **)&args, "ARGS", sizeof(DPU_LAUNCH_ARGS));
  }
}

void check_task(const Task *task, const std::vector<PhysicalRegion> &regions,
                Context ctx, Runtime *runtime) {
  assert(regions.size() == 2);
  assert(task->regions.size() == 2);
  assert(task->arglen == sizeof(TYPE));
  const TYPE alpha = *((const TYPE *)task->args);

  const AccessorRO acc_x(regions[0], FID_X);
  const AccessorRO acc_y(regions[0], FID_Y);
  const AccessorRO acc_z(regions[1], FID_Z);

  Rect<1> rect_xy = runtime->get_index_space_domain(
      ctx, task->regions[0].region.get_index_space());

  Rect<1> rect_z = runtime->get_index_space_domain(
      ctx, task->regions[1].region.get_index_space());

  int counter = 0;
  for(PointInRectIterator<1> pir_x(rect_xy); pir_x(); pir_x++){
    if(counter % WIDTH == 0) printf("\n");
    printf("%f ", acc_x[*pir_x]);
    counter++;
  }

  printf("\n");

  counter = 0;
  for(PointInRectIterator<1> pir_y(rect_xy); pir_y(); pir_y++){
    if(counter % WIDTH == 0) printf("\n");
    printf("%f ", acc_y[*pir_y]);
    counter++;
  }
  
  // const void *ptr = acc_z.ptr(rect_z.lo);
  // printf("Checking results... xptr %p, y_ptr %p, z_ptr %p...\n",
        //  acc_x.ptr(rect_xy.lo), acc_y.ptr(rect_xy.lo), ptr);
  bool all_passed = true;
  unsigned int count = 0;
  size_t errors = 0;
  unsigned int subregion_size = WIDTH*HEIGHT/(NUM_SUBREGIONS * NUM_SUBREGIONS);
  for (PointInRectIterator<1> pir(rect_z); pir(); pir++) {
    TYPE received = acc_z[*pir];
    TYPE expected = 0;

    int row = count / WIDTH;
    int col = count % WIDTH;

    PointInRectIterator<1> pir_x(rect_xy);
    PointInRectIterator<1> pir_y(rect_xy);

    for(int i=0; i<row*WIDTH; i++) pir_x++;
    for(int i=0; i<col*HEIGHT; i++) pir_y++;
    // pir_x += row*WIDTH;
    // pir_y += col*HEIGHT;

    for(int i=0; i<WIDTH; i++){
      // printf("(%f, %f) ", acc_x[*pir] ,acc_y[*pir]);

      expected += acc_x[*pir_x] * acc_y[*pir_y];
      pir_x++;
      pir_y++;
      // count++;
    }    


    // // printf("(%f, %f) ", acc_x[*pir] ,acc_y[*pir]);
    // TYPE received = acc_z[*pir];
    // TYPE expected = 0;

    // unsigned int subregion_index = count / (subregion_size);
    // unsigned int within_subregion_index = count % (subregion_size);
    // unsigned int subregion_width = WIDTH/NUM_SUBREGIONS;
    // unsigned int subregion_height = HEIGHT/NUM_SUBREGIONS;
    // unsigned int start_row = (subregion_index/NUM_SUBREGIONS) * HEIGHT;
    // unsigned int start_col = (subregion_index%NUM_SUBREGIONS) * subregion_width;


    // int row = start_row + within_subregion_index/subregion_width;
    // int col = start_col + within_subregion_index%subregion_width;

    // PointInRectIterator<1> pir_x(rect_xy);
    // PointInRectIterator<1> pir_y(rect_xy);

    // for(int i=0; i<row*WIDTH; i++) pir_x++;
    // for(int i=0; i<col*HEIGHT; i++) pir_y++;
    // // pir_x += row*WIDTH;
    // // pir_y += col*HEIGHT;

    // for(int i=0; i<WIDTH; i++){
    //   // printf("(%f, %f) ", acc_x[*pir] ,acc_y[*pir]);

    //   expected += acc_x[*pir_x] * acc_y[*pir_y];
    //   pir_x++;
    //   pir_y++;
    //   // count++;
    // }    
    // PRINT_EXPECTED(expected, received);
    // printf("location: %ld\n", count);
    // TYPE expected = alpha * acc_x[*pir] + acc_y[*pir];
    // Probably shouldn't check for floating point equivalence but
    // the order of operations are the same should they should
    // be bitwise equal.
    if (!COMPARE(expected, received)) {
      all_passed = false;
      // PRINT_EXPECTED(expected, received);
      // printf("location: %u\n", count);
      errors++;
    }
    count++;
    // count+=32;
  }
  if (all_passed)
    printf("SUCCESS!\n");
  else {
    printf("FAILURE!\n");
    printf("%ld ERRORS WERE FOUND\n", errors);
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
    TaskVariantRegistrar registrar(INIT_FIELD_TRANSPOSE_TASK_ID, "init_field_transpose");
    registrar.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
    registrar.set_leaf();
    Runtime::preregister_task_variant<init_field_task_transpose>(registrar, "init_field_transpose");
  }
  
  {
    TaskVariantRegistrar registrar(MULTIPLY_TASK_ID, "multiply");
    registrar.add_constraint(ProcessorConstraint(Processor::DPU_PROC));
    registrar.set_leaf();
    Runtime::preregister_task_variant<multiply_task>(registrar, "multiply");
  }

  {
    TaskVariantRegistrar registrar(CHECK_TASK_ID, "check");
    registrar.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
    registrar.set_leaf();
    Runtime::preregister_task_variant<check_task>(registrar, "check");
  }

  return Runtime::start(argc, argv);
}