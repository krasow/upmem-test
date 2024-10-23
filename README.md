## Fetch submodules
```bash
git submodule init 
git submodule update
source libs/download-upmem.sh
```
## ENV
```bash
source ENV
```
## Run
```bash
python run.py [BENCHMARK] [MODEL] --args [PROGRAM ARGS] --build_cmd [BUILD COMMAND]
python visual.py
```
### An example command:
```bash
python run.py daxby legion-pim --args "-ll:num_dpus 32 -b 32" --build_cmd "make -j"  
```

