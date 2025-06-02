CC = gcc
CFLAGS = -g -Wall -Werror -O2 -fno-tree-vectorize -fno-tree-loop-vectorize -fno-tree-slp-vectorize -fopenmp -march=native -DNOCUDA

NVCC = nvcc
NCFLAGS = -g -O2 -Xcompiler -fno-tree-vectorize -Xcompiler -fno-tree-loop-vectorize -Xcompiler -fno-tree-slp-vectorize

BUILD_DIR = ./build
BIN = $(BUILD_DIR)/test
NBIN = $(BUILD_DIR)/test_cuda

$(shell mkdir -p $(BUILD_DIR))

$(BIN): src/test.c src/baseline.c src/impl_row_noSIMD.c src/impl_col_noSIMD.c src/impl_col_SIMD.c
	@$(CC) $(CFLAGS) $^ -o $@

$(NBIN): src/test.c src/baseline.c src/impl_CUDA.cu
	@$(NVCC) $(NCFLAGS) $^ -o $@

run: $(BIN)	
	@$^

nrun: $(NBIN)
	@$^

clean: 
	-@rm -rf $(BUILD_DIR)

.PHONY: run nrun clean
