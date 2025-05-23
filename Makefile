CC = gcc
CFLAGS = -g -Wall -Werror -O2 -fno-tree-vectorize -fno-tree-loop-vectorize -fno-tree-slp-vectorize -fopenmp -march=native

BUILD_DIR = ./build
BIN = $(BUILD_DIR)/test

$(shell mkdir -p $(BUILD_DIR))

$(BIN): src/test.c src/baseline.c src/impl_row_noSIMD.c src/impl_col_noSIMD.c src/impl_col_SIMD.c src/impl.c
	@$(CC) $(CFLAGS) $^ -o $@

run: $(BIN)	
	@$^

clean: 
	-@rm -rf $(BUILD_DIR)

.PHONY: run clean
