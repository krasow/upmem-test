va: host.c
	@mkdir -p bin
	gcc --std=c99 -lm -fopenmp -O3 host.c -o bin/host -I$(SIMPLE_PIM_LIB) $(SIMPLE_PIM_LIB)/processing/ProcessingHelperHost.c  $(SIMPLE_PIM_LIB)/communication/CommHelper.c   $(SIMPLE_PIM_LIB)/communication/CommOps.c  $(SIMPLE_PIM_LIB)/management/SmallTableInit.c  $(SIMPLE_PIM_LIB)/management/Management.c  $(SIMPLE_PIM_LIB)/processing/map/Map.c  $(SIMPLE_PIM_LIB)/processing/zip/Zip.c `dpu-pkg-config --cflags --libs dpu`

clean:
	rm -rf bin
