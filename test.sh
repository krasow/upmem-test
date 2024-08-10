# make clean
nice make -j 100

./upmem_test -ll:num_dpus 1

python visual.py -d uint32 -i host_array.bin -o host_array.png -r 32 -c 32
python visual.py -d uint32 -i array.bin      -o device_array.png -r 32 -c 32
