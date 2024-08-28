size=64
type=int32

nice make -j 100

# for i in {1..10}
# do
#   ./upmem_test -ll:num_dpus 1

#   if [ $? -ne 0 ]; then
#     echo "Error occurred at iteration $i. Exiting loop."
#     break
#   fi
# done


./upmem_test -ll:num_dpus 1

python visual.py -d $type -i host_array.bin -o host_array.png -r $size -c $size 
python visual.py -d $type -i array.bin      -o device_array.png -r $size -c $size

diff host_array.png device_array.png
