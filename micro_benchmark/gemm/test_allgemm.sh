mkdir -p log
make allgemm
# Only run FP16 mul with FP16 accumulate on Tensorcore
./bin/allgemm FP16_TENSOR | tee log/allgemm_FP16_TENSOR.csv
# Only run FP16 mul with FP32 accumulate on Tensorcore
./bin/allgemm FP16_32_TENSOR | tee log/allgemm_FP16_32_TENSOR.csv
# Only run FP32 mul with FP32
./bin/allgemm FP32_CUDA | tee log/allgemm_FP32_CUDA.csv
# Only run FP16 mul with FP16
./bin/allgemm FP16_CUDA | tee log/allgemm_FP16_CUDA.csv
