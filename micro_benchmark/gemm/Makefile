#compilation option
NVCC=nvcc
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
	$(CUDA_PATH)/bin/$(NVCC) allgemm.cu -o $(BIN_DIR)/allgemm -lcublas -O3 -use_fast_math -Xcompiler -fPIC -std=c++11 -gencode=arch=compute_60,code=sm_60 -gencode=arch=compute_61,code=sm_61 -gencode=arch=compute_70,code=sm_70 -gencode=arch=compute_75,code=sm_75 -gencode=arch=compute_75,code=compute_75
	$(CUDA_PATH)/bin/$(NVCC) igemm_imma.cu -o $(BIN_DIR)/igemm_imma -lcublasLt -lcublas -O3 -use_fast_math -Xcompiler -fPIC -std=c++11 -gencode=arch=compute_60,code=sm_60 -gencode=arch=compute_61,code=sm_61 -gencode=arch=compute_70,code=sm_70 -gencode=arch=compute_75,code=sm_75 -gencode=arch=compute_75,code=compute_75


#clean
clean:
	rm -rf $(BIN_DIR)
#rebuild
rebuild: clean all
