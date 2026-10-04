NVCC   ?= nvcc
FLAGS  := -std=c++17 -O3 -lineinfo -arch=native

PROGS := device_info hello_gpu vector_add vector_benchmark vector_gridstride \
         reduce_sum matrix_mul matrix_mul_tiled vector_add_bug vector_add_fixed \
         reduce_sum_bug

all: $(PROGS)

%: %.cu cuda_check.cuh
	$(NVCC) $(FLAGS) $< -o $@

clean:
	rm -f $(PROGS)

.PHONY: all clean
