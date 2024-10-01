#include <stdio.h>
#include <assert.h>
#include <sys/mman.h>
#include <stdlib.h>
#include <stdint.h>
int add(int a, int b) {
    return a + b;
}

int subtract(int a, int b) {
    return a - b;
}

int main( void ) {
    int a = 5;
    int b = 3;
    printf("a+b = %d\n", add(a, b));

    unsigned long *add_func_ptr = (unsigned long *)add;
    unsigned long *subtract_func_ptr = (unsigned long *)subtract;

    if(mprotect((void *)(((uintptr_t)add_func_ptr / 4096) * 4096), 4096,  PROT_WRITE | PROT_READ | PROT_EXEC)) {
        perror("mprotect failed");
        exit(-1);
    }

    *(uint64_t *)add_func_ptr =  0xc300000000c0c748;


    // asm volatile (
    //     "mov %0, (%1)"
    //     :
    //     : "r"(subtract_func_ptr), "r"(add_func_ptr)
    // );

    printf("a-b = %d\n", add(a, b));
}
