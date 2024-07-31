#include <stdint.h>
#include <stdio.h>
#include <defs.h>
#include <mram.h>
#include <alloc.h>
#include <barrier.h>

int goat;

volatile void* something;
volatile void* two;

// Barrier
BARRIER_INIT(my_barrier, NR_TASKLETS);


int main(void) { 
    mem_reset();
    int dog;
    int cat;
    
    printf("dog %p\n", &dog); // WRAM stack
    printf("cat %p\n", &cat); // WRAM stack
    printf("goat %p\n", &goat); // WRAM global

    printf("main %p\n", &main); 
    
    double newval = 2.0000; // WRAM stack 
    for(int i = 1; i < 10; i++) {
        something = mem_alloc(i * 16); // WRAM heap

        ((volatile double*)something)[i] = 5.0000; // allocate on write? set value to allocation

        // read from WRAM and write to MRAM
        mram_write((const void *)(something),
                    (__mram_ptr void *)((uintptr_t)DPU_MRAM_HEAP_POINTER    +
                                        (uintptr_t)(i)), sizeof(double));

        printf("malloc(%d * 16) %p\n", i, something);
    }

    two = mem_alloc(1024);
    printf("malloc(1024) %p\n", two);

    return 0;
}