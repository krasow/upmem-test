#!/bin/bash
scripts=python_scripts
stderr=error.out
stdout=test.out
verbose=false  # or false

[ -f error.out ] && rm error.out
[ -f test.out ] && rm test.out

generate_param_file_simple_pim() {
    local dpus=$1
    local num_elems=$2
    local param_file="./daxby/simple-pim/Param.h"

    cat > "$param_file" <<EOF
#ifndef PARAM_H
#define PARAM_H
#include <stdint.h>
#include <stdlib.h>
typedef int32_t T; 
const uint32_t dpu_number = ${dpus};
uint32_t print_info = 0;
uint64_t nr_elements = ${num_elems};
#endif
EOF
}

i=1
fixed_exp=22  # Keep problem size constant
num_elems=$((2**fixed_exp))
dpus_list=(4 8 16 32 64)
trials=10
total=$(( ${#dpus_list[@]} * trials ))

for dpus in "${dpus_list[@]}"; do
  subregions=$dpus
  generate_param_file_simple_pim "$dpus" "$num_elems" 

  for trial in $(seq 1 $trials); do
    echo "DPUs: ${dpus} | Elements: ${num_elems} | Trial: ${trial}"
    
    CMD_LEGION=(
        python3 "$scripts/run.py" daxby legion-pim
        --args "-ll:num_dpus ${dpus} -b ${subregions} -n ${num_elems}"
        --build_cmd "make -j"
        --time_output "daxby_dpu${dpus}elem${num_elems}trial${trial}.out"
    )

    CMD_SIMPLEPIM=(
        python3 "$scripts/run.py" daxby simple-pim
        --run_cmd "./bin/host"
        --build_cmd "make clean && make -j"
        --time_output "daxby_dpu${dpus}elem${num_elems}trial${trial}.out"
    )

    if [ "$verbose" = true ]; then
      "${CMD_LEGION[@]}" 2>>"$stderr" | tee -a "$stdout"
      "${CMD_SIMPLEPIM[@]}" 2>>"$stderr" | tee -a "$stdout"
    else
      "${CMD_LEGION[@]}" >>"$stdout" 2>>"$stderr"
      "${CMD_SIMPLEPIM[@]}" >>"$stdout" 2>>"$stderr"
    fi
    i=$((i + 1))  

  done
done


echo "STDERR output below"
cat error.out
