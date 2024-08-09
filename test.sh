# make clean
make -j 100
chmod +x upmem_test

./upmem_test -ll:num_dpus 1

