#include <stdio.h>

#include <dpu>

#define TESTUP "test.bin"

int main(void) {

  struct dpu_set_t stream;
  DPU_ASSERT(dpu_alloc(64, NULL, &stream));

  DPU_ASSERT(dpu_load(stream, TESTUP, NULL));

  DPU_ASSERT(dpu_free(stream));
}
