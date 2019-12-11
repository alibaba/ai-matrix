#include <iostream>
#include <curand.h>
#include <cublas_v2.h>
#include <iomanip>
#include <vector>
#include <cstdlib>

#define MAX(x, y) ((x>y) ? x : y)
// Define some error checking macros.
#define cudaErrCheck(stat) { cudaErrCheck_((stat), __FILE__, __LINE__); }
void cudaErrCheck_(cudaError_t stat, const char *file, int line) {
	if (stat != cudaSuccess) {
		fprintf(stderr, "CUDA Error: %s %s %d\n", cudaGetErrorString(stat), file, line);
	}
}

#define cublasErrCheck(stat) { cublasErrCheck_((stat), __FILE__, __LINE__); }
void cublasErrCheck_(cublasStatus_t stat, const char *file, int line) {
	if (stat != CUBLAS_STATUS_SUCCESS) {
		fprintf(stderr, "cuBLAS Error: %d %s %d\n", stat, file, line);
	}
}

#define curandErrCheck(stat) { curandErrCheck_((stat), __FILE__, __LINE__); }
void curandErrCheck_(curandStatus_t stat, const char *file, int line) {
	if (stat != CURAND_STATUS_SUCCESS) {
		fprintf(stderr, "cuRand Error: %d %s %d\n", stat, file, line);
	}
}


double cal_tflops(int m, int n, int k, double msec)
{
    double flops = 2. * m * n * k;
    double tflops = (1E-12*flops) / (1E-3*msec);
    return tflops;
}

 

__global__ void assignFloatValue (float *out, int n, float value) {
	int idx = blockDim.x * blockIdx.x + threadIdx.x;
	if (idx < n) {
		out[idx] = value;
	}
}

__global__ void assignHalfValue (half *out, int n, float value) {
	int idx = blockDim.x * blockIdx.x + threadIdx.x;
	if (idx < n) {
		out[idx] = value;
	}
}
void correctnessCheck(int m, int n, int k, float *host, float value){
        for (int i = 0; i < m * n; i++) {      
            float val = host[i];
            if ( val != k * value * value) {
                std::cout << "ERROR value = " << val<< std::endl;
            }
        }
}

void printTime(float cublasTime, int m, int n, int k, float &s_max_tflops, int &s_max_m_n, int &s_max_k ){
        float tflops = cal_tflops(m, n, k, cublasTime);
        if (tflops > s_max_tflops){
            s_max_tflops = tflops;
	    s_max_m_n = m;
            s_max_k = k;
        }
        std::cout << std::setw(7) << m << ",";
        std::cout << std::setw(7) << n << ",";
        std::cout << std::setw(7) << k << ",";
        std::cout << std::setw(15) << std::setprecision(4) << cublasTime << ",";
        std::cout << std::setw(15) << std::setprecision(4) << tflops << "," << std::endl;
}

void calFP16Tensor(int m, int n, int k, float &s_max_tflops, int &s_max_m_n, int &s_max_k, int numRepeats){
        half *a_fp16;
        half *b_fp16;
        half *c_cublas;
        float *c_host_cublas;
        const float  value = 1.0f;
   
        cublasHandle_t cublasHandle;

        cudaEvent_t startcublas;
        cudaEvent_t stopcublas;

        cudaErrCheck(cudaEventCreate(&startcublas));
        cudaErrCheck(cudaEventCreate(&stopcublas));
        cublasErrCheck(cublasCreate(&cublasHandle));
        // Use tensor cores
        cublasErrCheck(cublasSetMathMode(cublasHandle, CUBLAS_TENSOR_OP_MATH));

        cudaErrCheck(cudaMalloc((void**)&a_fp16, m * k * sizeof(half)));
        cudaErrCheck(cudaMalloc((void**)&b_fp16, k * n * sizeof(half)));
        cudaErrCheck(cudaMalloc((void**)&c_cublas, m * n * sizeof(half)));
        c_host_cublas = (float*)malloc(m * n * sizeof(float));

        // curand doesn't currently support fp16 so we generate in fp32 and convert to fp16.
        assignHalfValue <<< (m * k + 255) / 256, 256 >>> (a_fp16, m*k, value);
        assignHalfValue <<< (k * n + 255) / 256, 256 >>> (b_fp16, k*n, value);
        assignHalfValue <<< (k * n + 255) / 256, 256 >>> (c_cublas, m*n, 0.0f);

        float alpha = 1.0f;
        float beta = 0.0f;

        // Now using cuBLAS
        cudaErrCheck(cudaEventRecord(startcublas));
        for (int iteration = 0; iteration < numRepeats; ++iteration) {
        cublasGemmEx(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_T, 
                    m, n, k, 
                    &alpha,
                    a_fp16, CUDA_R_16F, m,
                    b_fp16, CUDA_R_16F, n,
                    &beta, 
                    c_cublas, CUDA_R_16F, m,
                    CUDA_R_16F, CUBLAS_GEMM_DFALT_TENSOR_OP);
        }
        cudaErrCheck(cudaEventRecord(stopcublas));
        cudaErrCheck(cudaEventSynchronize(stopcublas));
        // TODO: Correctness check
        //cudaErrCheck(cudaMemcpy(c_host_cublas, c_cublas, m * n * sizeof(float), cudaMemcpyDeviceToHost));
        //correctnessCheck(m, n, k, c_host_cublas, value);
        // Check time
        float cublasTime;	
        cudaErrCheck(cudaEventElapsedTime(&cublasTime, startcublas, stopcublas)); 
        cublasTime /= numRepeats;
        printTime(cublasTime, m, n, k, s_max_tflops, s_max_m_n, s_max_k);
        
        cudaErrCheck(cudaEventDestroy(startcublas));             
        cudaErrCheck(cudaEventDestroy(stopcublas));
        cudaErrCheck(cudaFree(a_fp16));
        cudaErrCheck(cudaFree(b_fp16));
        cudaErrCheck(cudaFree(c_cublas));
        free(c_host_cublas);
}

void calFP16Accu32Tensor(int m, int n, int k, float &s_max_tflops, int &s_max_m_n, int &s_max_k, int numRepeats){
        half *a_fp16;
        half *b_fp16;
        float *c_cublas;
        float *c_host_cublas;
        const float  value = 1.0f;
        cublasHandle_t cublasHandle;
        cudaEvent_t startcublas;
        cudaEvent_t stopcublas;

        cudaErrCheck(cudaEventCreate(&startcublas));
        cudaErrCheck(cudaEventCreate(&stopcublas));
        cublasErrCheck(cublasCreate(&cublasHandle));
        // Use tensor cores
        cublasErrCheck(cublasSetMathMode(cublasHandle, CUBLAS_TENSOR_OP_MATH));

        cudaErrCheck(cudaMalloc((void**)&a_fp16, m * k * sizeof(half)));
        cudaErrCheck(cudaMalloc((void**)&b_fp16, k * n * sizeof(half)));
        cudaErrCheck(cudaMalloc((void**)&c_cublas, m * n * sizeof(float)));
        c_host_cublas = (float*)malloc(m * n * sizeof(float));

        // curand doesn't currently support fp16 so we generate in fp32 and convert to fp16.
        assignHalfValue <<< (m * k + 255) / 256, 256 >>> (a_fp16, m*k, value);
        assignHalfValue <<< (k * n + 255) / 256, 256 >>> (b_fp16, k*n, value);
        assignFloatValue <<< (k * n + 255) / 256, 256 >>> (c_cublas, m*n, 0.0f);

        float alpha = 1.0f;
        float beta = 0.0f;
        // Warp up not really needed
        // Now using cuBLAS
        cudaErrCheck(cudaEventRecord(startcublas));
        for (int iteration = 0; iteration < numRepeats; ++iteration) {
        cublasErrCheck(cublasGemmEx(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_T, 
                    m, n, k, 
                    &alpha,
                    a_fp16, CUDA_R_16F, m,
                    b_fp16, CUDA_R_16F, n,
                    &beta, 
                    c_cublas, CUDA_R_32F, m,
                    CUDA_R_32F, CUBLAS_GEMM_DFALT_TENSOR_OP));
        }
        cudaErrCheck(cudaEventRecord(stopcublas));
        cudaErrCheck(cudaEventSynchronize(stopcublas));
        // Correctness check
        cudaErrCheck(cudaMemcpy(c_host_cublas, c_cublas, m * n * sizeof(float), cudaMemcpyDeviceToHost));
        correctnessCheck(m, n, k, c_host_cublas, value);
        // Check time
        float cublasTime;	
        cudaErrCheck(cudaEventElapsedTime(&cublasTime, startcublas, stopcublas)); 
        cublasTime /= numRepeats;
        printTime(cublasTime, m, n, k, s_max_tflops, s_max_m_n, s_max_k);
        
        cudaErrCheck(cudaEventDestroy(startcublas));             
        cudaErrCheck(cudaEventDestroy(stopcublas));
        cudaErrCheck(cudaFree(a_fp16));
        cudaErrCheck(cudaFree(b_fp16));
        cudaErrCheck(cudaFree(c_cublas));
        free(c_host_cublas);
}

void calFP32CUDA(int m, int n, int k, float &s_max_tflops, int &s_max_m_n, int &s_max_k, int numRepeats){
        float *a_fp32;
        float *b_fp32;
        float *c_cublas;
        float *c_host_cublas;
        const float  value = 1.0f;
        cublasHandle_t cublasHandle;

        cudaEvent_t startcublas;
        cudaEvent_t stopcublas;

        cudaErrCheck(cudaEventCreate(&startcublas));
        cudaErrCheck(cudaEventCreate(&stopcublas));
        cublasErrCheck(cublasCreate(&cublasHandle));
        // No tensor cores
        cublasErrCheck(cublasSetMathMode(cublasHandle, CUBLAS_DEFAULT_MATH));

        cudaErrCheck(cudaMalloc((void**)&a_fp32, m * k * sizeof(float)));
        cudaErrCheck(cudaMalloc((void**)&b_fp32, k * n * sizeof(float)));
        cudaErrCheck(cudaMalloc((void**)&c_cublas, m * n * sizeof(float)));
        c_host_cublas = (float*)malloc(m * n * sizeof(float));

        // curand doesn't currently support fp16 so we generate in fp32 and convert to fp16.
        assignFloatValue <<< (m * k + 255) / 256, 256 >>> (a_fp32, m*k, value);
        assignFloatValue <<< (k * n + 255) / 256, 256 >>> (b_fp32, k*n, value);
        assignFloatValue <<< (k * n + 255) / 256, 256 >>> (c_cublas, m*n, 0.0f);

        float alpha = 1.0f;
        float beta = 0.0f;
        
        cudaErrCheck(cudaEventRecord(startcublas));
        for (int iteration = 0; iteration < numRepeats; ++iteration) {
        cublasSgemm(cublasHandle,
                CUBLAS_OP_N,
                CUBLAS_OP_T,
                m,
                n,
                k,
                &alpha,
                a_fp32, m,
                b_fp32, n,
                &beta,
                c_cublas, m);
        }
        cudaErrCheck(cudaEventRecord(stopcublas));
        cudaErrCheck(cudaEventSynchronize(stopcublas));
        // Correctness check
        cudaErrCheck(cudaMemcpy(c_host_cublas, c_cublas, m * n * sizeof(float), cudaMemcpyDeviceToHost));
        correctnessCheck(m, n, k, c_host_cublas, value);
        // Check time
        float cublasTime = 0.0f;	
        cudaErrCheck(cudaEventElapsedTime(&cublasTime, startcublas, stopcublas)); 
        cublasTime /= numRepeats;
        printTime(cublasTime, m, n, k, s_max_tflops, s_max_m_n, s_max_k);
        
        cudaErrCheck(cudaEventDestroy(startcublas));             
        cudaErrCheck(cudaEventDestroy(stopcublas));
        cudaErrCheck(cudaFree(a_fp32));
        cudaErrCheck(cudaFree(b_fp32));
        cudaErrCheck(cudaFree(c_cublas));
        free(c_host_cublas);
}


void calFP16CUDA(int m, int n, int k, float &s_max_tflops, int &s_max_m_n, int &s_max_k, int numRepeats){
        half *a_fp16;
        half *b_fp16;
        half *c_cublas;
        float *c_host_cublas;
        const float  value = 1.0f;
   
        cublasHandle_t cublasHandle;

        cudaEvent_t startcublas;
        cudaEvent_t stopcublas;

        cudaErrCheck(cudaEventCreate(&startcublas));
        cudaErrCheck(cudaEventCreate(&stopcublas));
        cublasErrCheck(cublasCreate(&cublasHandle));
        // No tensor cores
        cublasErrCheck(cublasSetMathMode(cublasHandle, CUBLAS_DEFAULT_MATH));

        cudaErrCheck(cudaMalloc((void**)&a_fp16, m * k * sizeof(half)));
        cudaErrCheck(cudaMalloc((void**)&b_fp16, k * n * sizeof(half)));
        cudaErrCheck(cudaMalloc((void**)&c_cublas, m * n * sizeof(half)));
        c_host_cublas = (float*)malloc(m * n * sizeof(float));

        // curand doesn't currently support fp16 so we generate in fp32 and convert to fp16.
        assignHalfValue <<< (m * k + 255) / 256, 256 >>> (a_fp16, m*k, value);
        assignHalfValue <<< (k * n + 255) / 256, 256 >>> (b_fp16, k*n, value);
        assignHalfValue <<< (k * n + 255) / 256, 256 >>> (c_cublas, m*n, 0.0f);

        half alpha = 1.0f;
        half beta = 0.0f;

        // Now using cuBLAS
        cudaErrCheck(cudaEventRecord(startcublas));
        for (int iteration = 0; iteration < numRepeats; ++iteration) {
        cublasHgemm(cublasHandle,
                CUBLAS_OP_N,
                CUBLAS_OP_T,
                m,
                n,
                k,
                &alpha,
                a_fp16, m,
                b_fp16, n,
                &beta,
                c_cublas, m);
        }
        cudaErrCheck(cudaEventRecord(stopcublas));
        cudaErrCheck(cudaEventSynchronize(stopcublas));
        // TODO: Correctness check
        //cudaErrCheck(cudaMemcpy(c_host_cublas, c_cublas, m * n * sizeof(float), cudaMemcpyDeviceToHost));
       //correctnessCheck(m, n, k, c_host_cublas, value);
        // Check time
        float cublasTime;	
        cudaErrCheck(cudaEventElapsedTime(&cublasTime, startcublas, stopcublas)); 
        cublasTime /= numRepeats;
        printTime(cublasTime, m, n, k, s_max_tflops, s_max_m_n, s_max_k);
        
        cudaErrCheck(cudaEventDestroy(startcublas));             
        cudaErrCheck(cudaEventDestroy(stopcublas));
        cudaErrCheck(cudaFree(a_fp16));
        cudaErrCheck(cudaFree(b_fp16));
        cudaErrCheck(cudaFree(c_cublas));
        free(c_host_cublas);
}
int main(int argc, char* argv[]) {
    int m,n,k;
    std::string precision="NULL";
    bool perf = true;
    if (argc < 3) {
        return EXIT_FAILURE;
    }
    
    // precision = INT8_TENSOR
    // precision = FP16_TENSOR
    // precision = FP16_32_TENSOR
    // precision = FP32_CUDA
    // precision = FP16_CUDA
    if (argc == 3) {
        precision = argv[1];
        std::string tmp = argv[2];
        if (tmp == "performance") perf= true;
	else if (tmp == "pressure") perf = false;
	else {
	  std::cout << "Invalid parameters!"<<std::endl;
	  return EXIT_FAILURE;
	}
    }
    
    float s_max_tflops = 0;
    int s_max_m_n = 0;
    int s_max_k = 0;
    int numRepeats;
/* // deprecated this INT8 test as it will achieve the best perf. Please refer to cublasLt    
    if (precision == "INT8_TENSOR" || precision == "NULL") {
    std::cout << "[TensorCore INT8(INT32 accumulation) Time and TOPS Result]" << std::endl;
    std::cout << std::setw(7) << "m" << std::setw(7) << "n" << std::setw(7) << "k";
    std::cout << std::setw(15) << "Time (msec)" << std::setw(15) << "TOPS";
    std::cout << std::endl;
    
    // for tensorcore test TODO: to verify the int8 with int8 accumulation
    for(m=1024, n = 1024; m <= 25600; m+=1024, n+=1024) {
    for(k=1024; k <= 20480; k+=1024) {
  
        int8_t *a_;
        int8_t *b_;
        int *c_cublas;
        int *c_host_cublas;
        //const int  value = 1;

   
        cublasHandle_t cublasHandle;

        cudaEvent_t startcublas;
        cudaEvent_t stopcublas;

        cudaErrCheck(cudaEventCreate(&startcublas));
        cudaErrCheck(cudaEventCreate(&stopcublas));
        cublasErrCheck(cublasCreate(&cublasHandle));
        // Use tensor cores
        cublasErrCheck(cublasSetMathMode(cublasHandle, CUBLAS_TENSOR_OP_MATH));

        cudaErrCheck(cudaMalloc((void**)&a_, m * k * sizeof(int8_t)));
        cudaErrCheck(cudaMalloc((void**)&b_, k * m * sizeof(int8_t)));
        cudaErrCheck(cudaMalloc((void**)&c_cublas, m * n * sizeof(int)));
        c_host_cublas = (int*)malloc(m * n * sizeof(int));

        //TODO curand doesn't currently support fp16 so we generate in fp32 and convert to fp16.
        //assignHalfValue <<< (m * k + 255) / 256, 256 >>> (a_fp16, m*k, value);
        //assignHalfValue <<< (k * n + 255) / 256, 256 >>> (b_fp16, k*n, value);
        //assignHalfValue <<< (k * n + 255) / 256, 256 >>> (c_cublas, m*n, 0.0f);

        int alpha = 1;
        int beta = 0;
        int numRepeats = 1;
        // Warp up not really needed here as many params will be tested
        // Now using cuBLAS
        cudaErrCheck(cudaEventRecord(startcublas));
        for (int iteration = 0; iteration < numRepeats; ++iteration) {
        cublasErrCheck(cublasGemmEx(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_T, 
                    m, n, k, 
                    &alpha,
                    a_, CUDA_R_8I, m,
                    b_, CUDA_R_8I, n,
                    &beta, 
                    c_cublas, CUDA_R_32I, m,
                    CUDA_R_32I, CUBLAS_GEMM_DFALT_TENSOR_OP));
        }
        cudaErrCheck(cudaEventRecord(stopcublas));
        cudaErrCheck(cudaEventSynchronize(stopcublas));
        // TODO: Correctness check
        //cudaErrCheck(cudaMemcpy(c_host_cublas, c_cublas, m * n * sizeof(float), cudaMemcpyDeviceToHost));
       //correctnessCheck(m, n, k, c_host_cublas, value);
        // Check time
        float cublasTime;	
        cudaErrCheck(cudaEventElapsedTime(&cublasTime, startcublas, stopcublas)); 
        cublasTime /= numRepeats;
        printTime(cublasTime, m, n, k, s_max_tflops, s_max_m_n, s_max_k);
        
        cudaErrCheck(cudaEventDestroy(startcublas));             
        cudaErrCheck(cudaEventDestroy(stopcublas));
        cudaErrCheck(cudaFree(a_));
        cudaErrCheck(cudaFree(b_));
        cudaErrCheck(cudaFree(c_cublas));
        free(c_host_cublas);
    }}
    std::cout << "[Peak TFLOPS]=" << s_max_tflops << ", m=n="<< s_max_m_n << ", k="<<s_max_k<< std::endl;
    cudaErrCheck(cudaDeviceReset());
    }
*/  

    //======= for tensorcore test
    // for perf test
    if (precision == "FP16_TENSOR" && perf == true) {
    std::cout << "[TensorCore FP16(FP16 accumulation) Time and TFLOPS Result]" << std::endl;
    std::cout << std::setw(7) << "m" << std::setw(7) << "n" << std::setw(7) << "k";
    std::cout << std::setw(15) << "Time (msec)" << std::setw(15) << "TFLOPS";
    std::cout << std::endl;
    s_max_tflops = 0;
    s_max_m_n = 0;
    s_max_k = 0;
    numRepeats = 10;
    for(m=1024, n = 1024; m <= 25600; m+=4096, n+=4096) {
    for(k=1024; k <= 20480; k+=4096) {
	calFP16Tensor( m, n, k,s_max_tflops, s_max_m_n, s_max_k, numRepeats);
    }}
    std::cout << "[Peak TFLOPS]=" << s_max_tflops << ", m=n="<< s_max_m_n << ", k="<<s_max_k<< std::endl;
    cudaErrCheck(cudaDeviceReset());
    }
    
    // for pressure test
    if (precision == "FP16_TENSOR" && perf == false) {
    std::cout << "[TensorCore FP16(FP16 accumulation) Time and TFLOPS Result]" << std::endl;
    std::cout << std::setw(7) << "m" << std::setw(7) << "n" << std::setw(7) << "k";
    std::cout << std::setw(15) << "Time (msec)" << std::setw(15) << "TFLOPS";
    std::cout << std::endl;
    s_max_tflops = 0;
    s_max_m_n = 0;
    s_max_k = 0;
    numRepeats = 2000;
    std::vector<int> mnk={512, 1024, 5120, 10240};
    for(int i=0; i<mnk.size(); i++) calFP16Tensor( mnk[i], mnk[i], mnk[i], s_max_tflops, s_max_m_n, s_max_k, numRepeats);
    
    cudaErrCheck(cudaDeviceReset());
    }
 
    // for perf test
    if (precision == "FP16_32_TENSOR" && perf == true) {
    std::cout << "[TensorCore FP16(FP32 accumulation) Time and TFLOPS Result]" << std::endl;
    std::cout << std::setw(7) << "m" << std::setw(7) << "n" << std::setw(7) << "k";
    std::cout << std::setw(15) << "Time (msec)" << std::setw(15) << "TFLOPS";
    std::cout << std::endl;
    s_max_tflops = 0;
    numRepeats = 10;
    for(m=1024, n = 1024; m <= 25600; m+=4096, n+=4096) {
    for(k=1024; k <= 20480; k+=4096) {
  	calFP16Accu32Tensor( m, n, k, s_max_tflops, s_max_m_n, s_max_k, numRepeats);
    }}
    std::cout << "[Peak TFLOPS]=" << s_max_tflops << ", m=n="<< s_max_m_n << ", k="<<s_max_k<< std::endl;
    cudaErrCheck(cudaDeviceReset());
    }

    // for pressure test
    if (precision == "FP16_32_TENSOR" && perf == false) {
    std::cout << "[TensorCore FP16(FP32 accumulation) Time and TFLOPS Result]" << std::endl;
    std::cout << std::setw(7) << "m" << std::setw(7) << "n" << std::setw(7) << "k";
    std::cout << std::setw(15) << "Time (msec)" << std::setw(15) << "TFLOPS";
    std::cout << std::endl;
    s_max_tflops = 0;
    numRepeats = 2000;
    std::vector<int> mnk={512, 1024, 5120, 10240};
    for(int i=0; i<mnk.size(); i++) calFP16Accu32Tensor( mnk[i], mnk[i], mnk[i], s_max_tflops, s_max_m_n, s_max_k, numRepeats);
    cudaErrCheck(cudaDeviceReset());
    }
    
    //======= for cudacore test
    if (precision == "FP32_CUDA" && perf == true) {
    std::cout << "[CUDA core FP32 Time and TFLOPS Result]" << std::endl;
    std::cout << std::setw(7) << "m" << std::setw(7) << "n" << std::setw(7) << "k";
    std::cout << std::setw(15) << "Time (msec)" << std::setw(15) << "TFLOPS";
    std::cout << std::endl;
    s_max_tflops = 0;
    numRepeats = 10;
    for(m=1024, n = 1024; m <= 21504; m+=4096, n+=4096) {
    for(k=1024; k <= 20480; k+=4096) {
	calFP32CUDA( m, n, k,s_max_tflops, s_max_m_n, s_max_k, numRepeats);
    }}
    std::cout << "[Peak TFLOPS]=" << s_max_tflops << ", m=n="<< s_max_m_n << ", k="<<s_max_k<< std::endl;
    cudaErrCheck(cudaDeviceReset());
    }
    // for pressure test
    if (precision == "FP32_CUDA" && perf == false) {
    std::cout << "[CUDA core FP32 Time and TFLOPS Result]" << std::endl;
    std::cout << std::setw(7) << "m" << std::setw(7) << "n" << std::setw(7) << "k";
    std::cout << std::setw(15) << "Time (msec)" << std::setw(15) << "TFLOPS";
    std::cout << std::endl;
    s_max_tflops = 0;
    numRepeats = 2000;
    std::vector<int> mnk={512, 1024, 5120, 10240};
    for(int i=0; i<mnk.size(); i++) calFP32CUDA( mnk[i], mnk[i], mnk[i], s_max_tflops, s_max_m_n, s_max_k, numRepeats);
    cudaErrCheck(cudaDeviceReset());
    }
    // for perf test
    if (precision == "FP16_CUDA" && perf == true) {
    std::cout << "[CUDA core FP16 Time and TFLOPS Result]" << std::endl;
    std::cout << std::setw(7) << "m" << std::setw(7) << "n" << std::setw(7) << "k";
    std::cout << std::setw(15) << "Time (msec)" << std::setw(15) << "TFLOPS";
    std::cout << std::endl;
    s_max_tflops = 0;
    numRepeats = 10;
    for(m=1024, n = 1024; m <= 21504; m+=4096, n+=4096) {
    for(k=1024; k <= 20480; k+=4096) {
	calFP16CUDA( m, n, k,s_max_tflops, s_max_m_n, s_max_k, numRepeats);
    }}
    std::cout << "[Peak TFLOPS]=" << s_max_tflops << ", m=n="<< s_max_m_n << ", k="<<s_max_k<< std::endl;
    cudaErrCheck(cudaDeviceReset());
    }
    // for pressure test
    if (precision == "FP16_CUDA" && perf == false) {
    std::cout << "[CUDA core FP16 Time and TFLOPS Result]" << std::endl;
    std::cout << std::setw(7) << "m" << std::setw(7) << "n" << std::setw(7) << "k";
    std::cout << std::setw(15) << "Time (msec)" << std::setw(15) << "TFLOPS";
    std::cout << std::endl;
    s_max_tflops = 0;
    numRepeats = 2000;
    std::vector<int> mnk={512, 1024, 5120, 10240};
    for(int i=0; i<mnk.size(); i++) calFP16CUDA( mnk[i], mnk[i], mnk[i], s_max_tflops, s_max_m_n, s_max_k, numRepeats);
    cudaErrCheck(cudaDeviceReset());
    }

    return 0;
}

