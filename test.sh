# make clean
nice make -j 100

for i in {1..10}
do
  ./upmem_test -ll:num_dpus 1

  if [ $? -ne 0 ]; then
    echo "Error occurred at iteration $i. Exiting loop."
    break
  fi
done


python visual.py -d uint32 -i host_array.bin -o host_array.png -r 32 -c 32
python visual.py -d uint32 -i array.bin      -o device_array.png -r 32 -c 32
diff host_array.png device_array.png
