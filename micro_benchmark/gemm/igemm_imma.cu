#include <iostream>
using namespace std;

#include <stdio.h>
#include <time.h>
#include <assert.h>
#include <stdlib.h>
#include <iomanip>
#include <cstdlib>
#include <vector>
// cuda
#include <cuda_runtime.h>
#include <cublas_v2.h>

#include <cublasLt.h>

#define Value   127
#define checkCudaAPIErrors(F) if ((F) != cudaSuccess) \
{ printf("Error at line %d in file %s: %s\n", __LINE__, __FILE__, cudaGetErrorString(cudaGetLastError())); exit(-1); }

static const char *_cudaGetErrorEnum(cublasStatus_t error)
{
    switch (error)
    {
        case CUBLAS_STATUS_SUCCESS:
            return "CUBLAS_STATUS_SUCCESS";

        case CUBLAS_STATUS_NOT_INITIALIZED:
            return "CUBLAS_STATUS_NOT_INITIALIZED";

        case CUBLAS_STATUS_ALLOC_FAILED:
            return "CUBLAS_STATUS_ALLOC_FAILED";

        case CUBLAS_STATUS_INVALID_VALUE:
            return "CUBLAS_STATUS_INVALID_VALUE";

        case CUBLAS_STATUS_ARCH_MISMATCH:
            return "CUBLAS_STATUS_ARCH_MISMATCH";

        case CUBLAS_STATUS_MAPPING_ERROR:
            return "CUBLAS_STATUS_MAPPING_ERROR";

        case CUBLAS_STATUS_EXECUTION_FAILED:
            return "CUBLAS_STATUS_EXECUTION_FAILED";

        case CUBLAS_STATUS_INTERNAL_ERROR:
            return "CUBLAS_STATUS_INTERNAL_ERROR";

        case CUBLAS_STATUS_NOT_SUPPORTED:
            return "CUBLAS_STATUS_NOT_SUPPORTED";

        case CUBLAS_STATUS_LICENSE_ERROR:
            return "CUBLAS_STATUS_LICENSE_ERROR";
    }

    return "<unknown>";
}

#define checkcuBlasError(F) if ((F) != CUBLAS_STATUS_SUCCESS) \
{ printf("Error at line %d in file %s: %s\n", __LINE__, __FILE__, _cudaGetErrorEnum(F)); exit(-1); }

int roundoff(int v, int d)
{
    return ((v+d-1)/d) * d;
}

__global__ void memKernel(int8_t * in, int8_t val, int size) {
    for(int i = 0; i < size; i++)
            in[i] = val;
}

int ltIgemmTensor(cublasLtHandle_t ltHandle,
        int m, int n, int k,
        const int8_t *A,
        int lda,
        const int8_t *B,
        int ldb,
        int32_t *C,
        int ldc,
        int iters,
        float &time_matmul)
{
    cublasStatus_t cublasStat = CUBLAS_STATUS_SUCCESS;

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cublasLtMatmulDesc_t matmulDesc;
    cublasLtMatrixLayout_t Adesc = NULL;
    cublasLtMatrixLayout_t Bdesc = NULL;
    cublasLtMatrixLayout_t Cdesc = NULL;

    int32_t alpha = 1;
    int32_t beta  = 0;

    cublasOperation_t opTranspose = CUBLAS_OP_T;

    // The tensor op igemm kernels require specialized memory order of data
    cublasLtMatrixTransformDesc_t transformDesc = NULL;
    int8_t  *Atransform = NULL;
    int8_t  *Btransform = NULL;
    int32_t *Ctransform = NULL;

    cublasLtMatrixLayout_t AtransformDesc = NULL;
    cublasLtMatrixLayout_t BtransformDesc = NULL;
    cublasLtMatrixLayout_t CtransformDesc = NULL;

    float transformAlpha = 1.0f;
    float transformBeta  = 0.0f;

    cublasLtOrder_t order_COL32 = CUBLASLT_ORDER_COL32;
    cublasLtOrder_t order_COL4_4R2_8C = CUBLASLT_ORDER_COL4_4R2_8C;

    int ldaTransform = 32 * m;
    int ldbTransform = 32 * roundoff(n,8);
    int ldcTransform = 32 * m;

    checkCudaAPIErrors(cudaMalloc((void **)&Atransform, sizeof(int8_t ) * roundoff(k, 32)/32*ldaTransform));
    checkCudaAPIErrors(cudaMalloc((void **)&Btransform, sizeof(int8_t ) * roundoff(k, 32)/32*ldbTransform));
    checkCudaAPIErrors(cudaMalloc((void **)&Ctransform, sizeof(int32_t )* roundoff(n, 32)/32*ldcTransform));

    // create transformDesc
    checkcuBlasError(cublasLtMatrixTransformDescCreate(&transformDesc, CUDA_R_32F));

    // create matmulDesc
    checkcuBlasError(cublasLtMatmulDescCreate(&matmulDesc, CUDA_R_32I));

    // Tensor op igemm kernels only support NT gemm
    checkcuBlasError(cublasLtMatmulDescSetAttribute(matmulDesc, CUBLASLT_MATMUL_DESC_TRANSB, &opTranspose, sizeof(cublasOperation_t)));

    // Create descriptors for the original matrices
    checkcuBlasError(cublasLtMatrixLayoutCreate(&Adesc, CUDA_R_8I,  m, k, lda));
    // B or transposed B
    checkcuBlasError(cublasLtMatrixLayoutCreate(&Bdesc, CUDA_R_8I,  n, k, ldb)); 
    checkcuBlasError(cublasLtMatrixLayoutCreate(&Cdesc, CUDA_R_32I, m, n, ldc)); 

    // Create descriptors for the transformed matrices
    checkcuBlasError(cublasLtMatrixLayoutCreate(&AtransformDesc, CUDA_R_8I,  m, k, ldaTransform));
    checkcuBlasError(cublasLtMatrixLayoutSetAttribute(AtransformDesc, CUBLASLT_MATRIX_LAYOUT_ORDER,  &order_COL32, sizeof(order_COL32)));
     
    checkcuBlasError(cublasLtMatrixLayoutCreate(&BtransformDesc, CUDA_R_8I,  n, k, ldbTransform));
    checkcuBlasError(cublasLtMatrixLayoutSetAttribute(BtransformDesc, CUBLASLT_MATRIX_LAYOUT_ORDER,  &order_COL4_4R2_8C, sizeof(order_COL4_4R2_8C)));

    checkcuBlasError(cublasLtMatrixLayoutCreate(&CtransformDesc, CUDA_R_32I, m, n, ldcTransform));
    checkcuBlasError(cublasLtMatrixLayoutSetAttribute(CtransformDesc, CUBLASLT_MATRIX_LAYOUT_ORDER,  &order_COL32, sizeof(order_COL32)));

    cublasLtMatmulPreference_t preference = NULL;
 
    int returnedResults = 0;
    cublasLtMatmulHeuristicResult_t heuristicResult = {};
    checkcuBlasError(cublasLtMatmulPreferenceCreate(&preference));
  
    checkcuBlasError(cublasLtMatmulAlgoGetHeuristic(
        ltHandle, matmulDesc, AtransformDesc, BtransformDesc, CtransformDesc, CtransformDesc, preference, 1, &heuristicResult, &returnedResults));
 
    cublasLtMatmulTile_t tileSize = CUBLASLT_MATMUL_TILE_128x256;
    cublasLtMatmulAlgoConfigSetAttribute( &heuristicResult.algo,  CUBLASLT_ALGO_CONFIG_TILE_ID, &tileSize, sizeof(cublasLtMatmulTile_t));
    if (returnedResults == 0) {
        checkcuBlasError(CUBLAS_STATUS_NOT_SUPPORTED);
    }

    checkcuBlasError(cublasLtMatrixTransform(ltHandle, 
                transformDesc, 
                &transformAlpha,
                A, Adesc,
                &transformBeta,
                NULL, NULL, 
                Atransform, AtransformDesc, 0));

    checkcuBlasError(cublasLtMatrixTransform(ltHandle, 
                transformDesc, 
                &transformAlpha,
                B, Bdesc,
                &transformBeta,
                NULL, NULL, 
                Btransform, BtransformDesc, 0));

    //for(int i = 0; i < 1; i++) {
    //        memKernel<<<1,1,0,0>>>(Atransform, 0, 2000);
    //    }

    cudaEventRecord(start, 0);
    for (int i=0; i<iters; i++)
    {
        checkcuBlasError(cublasLtMatmul(ltHandle, 
                    matmulDesc,
                    &alpha,
                    Atransform,
                    AtransformDesc,
                    Btransform,
                    BtransformDesc,
                    &beta,
                    Ctransform,
                    CtransformDesc,
                    Ctransform,
                    CtransformDesc,
                    &heuristicResult.algo, 
                    NULL, 0, 0));

    }
    cudaEventRecord(stop, 0);

    checkcuBlasError(cublasLtMatrixTransform(ltHandle, 
                transformDesc,
                &transformAlpha,
                Ctransform,
                CtransformDesc, 
                &transformBeta,
                NULL, NULL, 
                C, Cdesc, 0));

    //cudaEventSynchronize(stop);
    cudaDeviceSynchronize();
    time_matmul=0.0;
    cudaEventElapsedTime(&time_matmul, start, stop);

    time_matmul /= iters;

    // Descriptors are no longer needed as all GPU work was already enqueued.
    if (CtransformDesc) cublasLtMatrixLayoutDestroy(CtransformDesc);
    if (BtransformDesc) cublasLtMatrixLayoutDestroy(BtransformDesc);
    if (AtransformDesc) cublasLtMatrixLayoutDestroy(AtransformDesc);
    if (Cdesc) cublasLtMatrixLayoutDestroy(Cdesc);
    if (Bdesc) cublasLtMatrixLayoutDestroy(Bdesc);
    if (Adesc) cublasLtMatrixLayoutDestroy(Adesc);
    if (matmulDesc) cublasLtMatmulDescDestroy(matmulDesc);
    if (transformDesc) cublasLtMatrixTransformDescDestroy(transformDesc);

    // Wait until device is done before freeing transformed buffers
    cudaDeviceSynchronize();
    if (Ctransform) cudaFree(Ctransform);
    if (Btransform) cudaFree(Btransform);
    if (Atransform) cudaFree(Atransform);

    return cublasStat == CUBLAS_STATUS_SUCCESS ? 0 : 1;
}

// initialize matrix in column-major
void matInit(int rows, int cols, int8_t *p, int ld)
{
    srand(time(NULL)); 

    for (int c=0; c<cols; c++)
    {
        for (int r=0; r<rows; r++)
        {
            int index = r + c * ld;
            
            p[index] = rand()%10;
            //p[index] = 0;
        }
    }
}

void matDisplay(int rows, int cols, int8_t *p, int ld)
{
    for (int c=0; c<cols; c++)
    {
        for (int r=0; r<rows; r++)
        {
            int index = r + c * ld;

            printf("%4d", p[index]);    
        }
    }
}

// mat  is column-major
// matT is row-major 
void transpose(int8_t *matT, int8_t *mat, int rows, int cols)
{
    for (int c=0; c<cols; c++)
    {
        for (int r=0; r<rows; r++)
        {
            int indexIn = r + c*rows;
            int indexOut= c + r*cols;

            matT[indexOut] = mat[indexIn];
        }
    }
}

void matMul(int m, int n, int k,
        const int8_t *A,
        int lda,
        const int8_t *B,
        int ldb,
        int32_t *C,
        int ldc)
{
    int32_t sum;

    for (int c=0; c<n; c++)
    {
        for (int r=0; r<m; r++)
        {
            sum = 0; 

            for (int kk=0; kk<k; kk++)
            {
                int idxA = kk*lda + r; // A[r][kk]       
                int idxB = c*ldb + kk; // B[kk][c]

                sum += A[idxA] * B[idxB];
            }

            C[c*ldc + r] = sum; // C[r][c]
        }
    }
}

void postprocess(const int32_t *ref, const int32_t *res, int m, int n, int k, float ms)
{

    for (int c=0; c<n; c++)
    {
        for (int r=0; r<m; r++)
        {
            int index = r + c*m;

            if (ref[index] !=  res[index])
            {
                printf("(row = %d, col = %d) gpu result=%d cpu ref=%d  ", r, c, res[index], ref[index]);
                printf("%25s\n", "*** FAILED ***");
                break;
            }
        }
    }

}

double cal_tflops(int m, int n, int k, double msec)
{   
    double flops = 2. * m * n * k;
    double tflops = (1E-12*flops) / (1E-3*msec);
    return tflops;
}

void printTime(float cublasTime, int m, int n, int k, float &s_max_tflops, int &s_max_m_n, int &s_max_k ){
        float tflops = cal_tflops(m, n, k, cublasTime);
        if (tflops > s_max_tflops){
            s_max_tflops = tflops;
            s_max_m_n = m;
            s_max_k = k;
        }
        cout << setw(7) << m << ",";
        cout << setw(7) << n << ",";
        cout << setw(7) << k << ",";
        cout << setw(15) << setprecision(4) << cublasTime << ",";
        cout << setw(15) << setprecision(4) << tflops << "," << endl;
}

void calINT8Accu32Tensor(int m, int n, int k, float &s_max_tflops, int &s_max_m_n, int &s_max_k, int iters){


    int devID = 0;

    int8_t  *h_A = NULL; // m * k, stored in column-major
    int8_t  *h_B = NULL; // k * n, stored in column-major
    int8_t  *h_BT = NULL; // k * n, stored in column-major
    int32_t *h_C = NULL; // m * n, stored in column-major
    int32_t *h_Cres = NULL; // m * n, stored in column-major

    int8_t  *d_A = NULL; // m * k, stored in column-major
    int8_t  *d_B = NULL; // k * n, stored in column-major
    int8_t  *d_BT= NULL; // k * n, stored in column-major
    int32_t *d_C = NULL; // m * n, stored in column-major

    // allocate memory
    h_A = (int8_t* )malloc(sizeof(int8_t ) * m * k);
    if (!h_A) printf("falied to allocate mem on CPU");
    h_B = (int8_t* )malloc(sizeof(int8_t ) * k * n);   // B : k*n
    if (!h_B) printf("falied to allocate mem on CPU"); // BT: n*k, the transpose of B
    h_BT= (int8_t* )malloc(sizeof(int8_t ) * n * k);
    if (!h_BT) printf("falied to allocate mem on CPU");
    h_C = (int32_t*)malloc(sizeof(int32_t) * m * n);
    if (!h_C) printf("falied to allocate mem on CPU");
    h_Cres = (int32_t*)malloc(sizeof(int32_t) * m * n);
    if (!h_Cres) printf("falied to allocate mem on CPU");

    checkCudaAPIErrors(cudaMalloc((void **)&d_A, sizeof(int8_t ) * m * k));
    checkCudaAPIErrors(cudaMalloc((void **)&d_B, sizeof(int8_t ) * k * n));
    checkCudaAPIErrors(cudaMalloc((void **)&d_BT,sizeof(int8_t ) * n * k));
    checkCudaAPIErrors(cudaMalloc((void **)&d_C, sizeof(int32_t) * m * n));

    cudaSetDevice(devID);
    cudaDeviceProp devProp;
    cudaGetDeviceProperties(&devProp, devID);
    //printf("Device : %s, compute SM %d.%d.\n",devProp.name, devProp.major, devProp.minor);

    cublasLtHandle_t ltHandle;
    checkcuBlasError(cublasLtCreate(&ltHandle));

    cublasHandle_t handle;
    checkcuBlasError(cublasCreate(&handle));

    float time_matmul= 0.0; // ms

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);


    // step 1: initialize A and B
    //printf("step 1: initialize A and B with m=%d, n=%d and k=%d\n", m, n, k);
    matInit(m, k, h_A, m);
    //matDisplay(m, k, h_A, m);
    matInit(k, n, h_B, k);
    transpose(h_BT, h_B, k, n);
    //matDisplay(k, n, h_B, k);

    // step 2: compute matmul on cpu  
    //printf("step 2: do gemm on CPU\n");
    //matMul(m, n, k, h_A, m, h_B, k, h_C, m);

    //step 3: copy date from host to device
    //printf("step 3: copy date from host to device\n");
    checkCudaAPIErrors(cudaMemcpy(d_A, h_A, sizeof(int8_t) * m * k,cudaMemcpyHostToDevice));
    checkCudaAPIErrors(cudaMemcpy(d_B, h_B, sizeof(int8_t) * k * n,cudaMemcpyHostToDevice));
    checkCudaAPIErrors(cudaMemcpy(d_BT,h_BT,sizeof(int8_t) * n * k,cudaMemcpyHostToDevice));
    cudaDeviceSynchronize();
/*
    // step 4-1: cublasGemmEx
    printf("step 4-1: call API cublasGemmEx\n");
    cudaEventRecord(start, 0);

    cublasStatus_t cublasStat;
    for (int t = 0; t < iters; t++)
    {
        cublasStat=cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N, 
                                m, n, k, 
                                &alpha, 
                                d_A, CUDA_R_8I, m, 
                                d_B, CUDA_R_8I, k, 
                                &beta, 
                                d_C, CUDA_R_32I, m,
                                CUDA_R_32I,             
                static_cast<cublasGemmAlgo_t>(CUBLAS_GEMM_DEFAULT));

        if(cublasStat != CUBLAS_STATUS_SUCCESS)
        {
            checkcuBlasError(cublasStat);
            continue;
        }
    }
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time_used, start, stop);

    time_used /= iters;

    checkCudaAPIErrors(cudaMemcpy(h_Cres, d_C, sizeof(int32_t) * m * n, cudaMemcpyDeviceToHost));
    postprocess(h_C, h_Cres, m, n, k, time_used);
*/
    // step 4-2: cublasLtMatMul
    //printf("step 4: call API cublasLtMatMul\n");

    ltIgemmTensor(ltHandle,
                  m, n, k,
                  d_A, m,
                  d_BT,n,
                  d_C, m,
                  iters,
                  time_matmul);

    checkCudaAPIErrors(cudaMemcpy(h_Cres, d_C, sizeof(int32_t) * m * n, cudaMemcpyDeviceToHost));
    // comment out the results check. It is verified before I upload
    //postprocess(h_C, h_Cres, m, n, k, time_matmul);

    printTime(time_matmul, m, n, k, s_max_tflops, s_max_m_n, s_max_k);


    //free memory
    free(h_A);
    free(h_B);
    free(h_BT);
    free(h_C);
    free(h_Cres);
    checkCudaAPIErrors(cudaFree(d_A));
    checkCudaAPIErrors(cudaFree(d_B));
    checkCudaAPIErrors(cudaFree(d_BT));
    checkCudaAPIErrors(cudaFree(d_C));

    checkCudaAPIErrors(cudaEventDestroy(start));
    checkCudaAPIErrors(cudaEventDestroy(stop));
    checkcuBlasError(cublasDestroy(handle));
    checkcuBlasError(cublasLtDestroy(ltHandle));
}


int main(int argc, char** argv)
{
    int m,n,k;
    bool perf = true;
    if (argc < 2) {
        return EXIT_FAILURE;
    }
    
    if (argc == 2) {
        std::string tmp = argv[1];
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

     // for perf test
    if (perf == true) {
        cout << "[TensorCore INT8(INT32 accumulation) Time and TOPS Result]" << std::endl;
        cout << setw(7) << "m" << setw(7) << "n" << setw(7) << "k";
        cout << setw(20) << "Time (msec)" << setw(15) << "TOPS";
        cout << endl;
        s_max_tflops = 0;
        numRepeats = 10;
        for(m=1024, n = 1024; m <= 25600; m+=4096, n+=4096) {
        for(k=1024; k <= 20480; k+=4096) {
            calINT8Accu32Tensor( m, n, k, s_max_tflops, s_max_m_n, s_max_k, numRepeats);
        }}
            
        cout << "[Peak TOPS]=" << s_max_tflops << ", m=n="<< s_max_m_n << ", k="<<s_max_k<< endl;
        checkCudaAPIErrors(cudaDeviceReset());
    }


    if (perf == false) {
        cout << "[TensorCore INT8(INT32 accumulation) Time and TOPS Result]" << std::endl;
        cout << setw(7) << "m" << setw(7) << "n" << setw(7) << "k";
        cout << setw(20) << "Time (msec)" << setw(15) << "TOPS";
        cout << endl;
        s_max_tflops = 0;
        numRepeats = 2000;
        std::vector<int> mnk={512, 1024, 5120, 10240};
        for(int i=0; i<mnk.size(); i++) calINT8Accu32Tensor( mnk[i], mnk[i], mnk[i], s_max_tflops, s_max_m_n, s_max_k, numRepeats);

        checkCudaAPIErrors(cudaDeviceReset());
    }
}

