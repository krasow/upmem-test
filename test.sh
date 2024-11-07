#!/bin/bash
scripts=python_scripts

# https://www.baeldung.com/linux/command-line-progress-bar

bar_size=40
bar_char_done="#"
bar_char_todo="-"
bar_percentage_scale=2

function show_progress {
  current="$1"
  total="$2"

  percent=$(bc <<< "scale=$bar_percentage_scale; 100 * $current / $total" )
  done=$(bc <<< "scale=0; $bar_size * $percent / 100" )
  todo=$(bc <<< "scale=0; $bar_size - $done" )
  # build the done and todo sub-bars
  done_sub_bar=$(printf "%${done}s" | tr " " "${bar_char_done}")
  todo_sub_bar=$(printf "%${todo}s" | tr " " "${bar_char_todo}")
  echo -ne "\rProgress : [${done_sub_bar}${todo_sub_bar}] ${percent}%"

  if [ $total -eq $current ]; then
    echo -e "\nDONE"
  fi
}



i=0
for dpus in 4 8 16 32; do
  subregions=$dpus
  for exp in {15..20}; do
    show_progress $i 23
    num_elems=$((2**exp))
    python3 $scripts/run.py daxby legion-pim --args "-ll:num_dpus ${dpus} -b ${subregions} -n ${num_elems}" --build_cmd "make -j" --time_output "daxby_dpu${dpus}elem${num_elems}.out" >> test.out 2>> error.out
    
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
    
  python3 $scripts/run.py daxby simple-pim --run_cmd "./bin/host" --build_cmd "make clean && make -j" --time_output "daxby_dpu${dpus}elem${num_elems}.out" >> test.out 2>> error.out
  i=$((i + 1))  
done
done

echo "STDERR output below"
cat error.out