# make clean
size=64
type=double

nice make -j 100

./upmem_test -ll:num_dpus 1

python visual.py -d $type -i host_array.bin -o host_array.png -r $size -c $size 
python visual.py -d $type -i array.bin      -o device_array.png -r $size -c $size
