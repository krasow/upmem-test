#!/bin/bash
scripts=python_scripts
for dpus in 4 8 16 32; do
  subregions=$dpus
  for exp in {10..20}; do
    num_elems=$((2**exp))
    python3 $scripts/run.py daxby legion-pim --args "-ll:num_dpus ${dpus} -b ${subregions} -n ${num_elems}" --build_cmd "make -j" --time_output "daxby_dpu${dpus}elem${num_elems}.out"
    
cat > ./daxby/simple-pim/Param.h <<EOF
    #ifndef PARAM_H
    #define PARAM_H
    #include <stdlib.h>
    typedef double T; 
    const uint32_t dpu_number = ${dpus};
    uint32_t print_info = 0;
    uint64_t nr_elements = ${num_elems};
    #endif
EOF
    
  python3 $scripts/run.py daxby simple-pim --run_cmd "./bin/host" --build_cmd "make clean && make -j" --time_output "daxby_dpu${dpus}elem${num_elems}.out"
  done
done
