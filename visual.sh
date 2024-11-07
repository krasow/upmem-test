#!/bin/bas
scripts=python_scripts
python3 $scripts/combine.py "./daxby/simple-pim/" "simple_pim_daxby.csv"
python3 $scripts/combine.py "./daxby/legion-pim/" "legion_daxby.csv"
python3 $scripts/visual.py "daxby/simple-pim/simple_pim_daxby.csv" "daxby/simple-pim/simple_pim_results.png"
python3 $scripts/visual.py "daxby/legion-pim/legion_daxby.csv" "daxby/legion-pim/legion_results.png"

