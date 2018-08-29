#compilation option
NVCC=nvcc
ARCH=sm_52
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
.PHONY=all sgemm dgemm
all: sgemm dgemm

sgemm:
	$(MKDIR) $(BIN_DIR) 
	$(CUDA_PATH)/bin/$(NVCC) sgemm.cu -o $(BIN_DIR)/sgemm -I $(CUDA_PATH)/include -L $(BLAS_PATH) -l$(BLAS_LIBRARY) -L $(CUDA_LIB64) -lcurand -arch=$(ARCH) -std=c++11

dgemm:
	$(MKDIR) $(BIN_DIR) 
	$(CUDA_PATH)/bin/$(NVCC) dgemm.cu -o $(BIN_DIR)/dgemm -I $(CUDA_PATH)/include -L $(BLAS_PATH) -l$(BLAS_LIBRARY) -L $(CUDA_LIB64) -lcurand -arch=$(ARCH) -std=c++11

#clean
clean:
	rm -rf $(BIN_DIR)
#rebuild
rebuild: clean all
