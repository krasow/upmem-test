#!/bin/bas
scripts=python_scripts
python3 $scripts/combine.py "./daxby/simple-pim/" "simple_pim_daxby.csv"
python3 $scripts/combine.py "./daxby/legion-pim/" "legion_daxby.csv"
python3 $scripts/visual.py "simple_pim" "daxby/simple-pim/simple_pim_daxby.csv" "daxby/simple-pim/simple_pim_results.png"
python3 $scripts/visual.py "legion_pim" "daxby/legion-pim/legion_daxby.csv" "daxby/legion-pim/legion_results.png"

