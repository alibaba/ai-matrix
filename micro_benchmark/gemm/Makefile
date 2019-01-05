#compilation option
NVCC=nvcc
ARCH=sm_60
#library path
CUDA_PATH?=/usr/local/cuda
CUDA_LIB64=$(CUDA_PATH)/lib64
#BLAS
BLAS_PATH?=$(CUDA_LIB64)
BLAS_LIBRARY?=cublas
#create directory
BIN_DIR?=bin
MKDIR=mkdir -p
#target
.PHONY=all allgemm
all: allgemm


allgemm:
	$(MKDIR) $(BIN_DIR) 
	$(CUDA_PATH)/bin/$(NVCC) allgemm.cu -o $(BIN_DIR)/allgemm -lcublas -O3 -use_fast_math -Xcompiler -fPIC -std=c++11


#clean
clean:
	rm -rf $(BIN_DIR)
#rebuild
rebuild: clean all
