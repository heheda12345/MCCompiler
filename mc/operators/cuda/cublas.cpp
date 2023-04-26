#include <curand.h>
#include <iomanip>
#include <sstream>

#include <vector>
#include <assert.h>
#include <iostream>
#include "common.h"
#include "cublas_utils.h"

inline void __curandSafeCall(curandStatus_t err, const char *file, const int line )
    {
    if( CURAND_STATUS_SUCCESS != err) {
        fprintf(stderr, "%s(%i) : curandSafeCall() CURAND error %d: ",
                file, line, (int)err);
        switch (err) {
            case CURAND_STATUS_VERSION_MISMATCH:    fprintf(stderr, "CURAND_STATUS_VERSION_MISMATCH");
            case CURAND_STATUS_NOT_INITIALIZED:     fprintf(stderr, "CURAND_STATUS_NOT_INITIALIZED");
            case CURAND_STATUS_ALLOCATION_FAILED:   fprintf(stderr, "CURAND_STATUS_ALLOCATION_FAILED");
            case CURAND_STATUS_TYPE_ERROR:          fprintf(stderr, "CURAND_STATUS_TYPE_ERROR");
            case CURAND_STATUS_OUT_OF_RANGE:        fprintf(stderr, "CURAND_STATUS_OUT_OF_RANGE");
            case CURAND_STATUS_LENGTH_NOT_MULTIPLE: fprintf(stderr, "CURAND_STATUS_LENGTH_NOT_MULTIPLE");
            case CURAND_STATUS_DOUBLE_PRECISION_REQUIRED: 
                                                    fprintf(stderr, "CURAND_STATUS_DOUBLE_PRECISION_REQUIRED");
            case CURAND_STATUS_LAUNCH_FAILURE:      fprintf(stderr, "CURAND_STATUS_LAUNCH_FAILURE");
            case CURAND_STATUS_PREEXISTING_FAILURE: fprintf(stderr, "CURAND_STATUS_PREEXISTING_FAILURE");
            case CURAND_STATUS_INITIALIZATION_FAILED:     
                                                    fprintf(stderr, "CURAND_STATUS_INITIALIZATION_FAILED");
            case CURAND_STATUS_ARCH_MISMATCH:       fprintf(stderr, "CURAND_STATUS_ARCH_MISMATCH");
            case CURAND_STATUS_INTERNAL_ERROR:      fprintf(stderr, "CURAND_STATUS_INTERNAL_ERROR");
            default: fprintf(stderr, "CURAND Unknown error code\n");
        }
        exit(-1);
    }
}

#define curandSafeCall(err) __curandSafeCall (err, __FILE__, __LINE__)

namespace MCCompiler {

constexpr int numWarmup = 32;
constexpr int numEval = 128;
constexpr bool enableVerification = true;

std::vector<cublasLtMatmulAlgo_t> getAlgo(const cublasLtHandle_t &handle,
                                          const cublasLtMatmulDesc_t &desc,
                                          const cublasLtMatrixLayout_t &layoutA,
                                          const cublasLtMatrixLayout_t &layoutB,
                                          const cublasLtMatrixLayout_t &layoutC,
                                          const cublasLtMatrixLayout_t &layoutD,
                                          size_t wsSize) {
    cublasLtMatmulPreference_t preference = nullptr;
    checkBlasErrors(cublasLtMatmulPreferenceCreate(&preference));
    checkBlasErrors(cublasLtMatmulPreferenceSetAttribute(
        preference, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &wsSize,
        sizeof(wsSize)));
    const int requestedAlgoCount = 8;
    int returnedAlgoCounts = 0;
    cublasLtMatmulHeuristicResult_t heuristicResult[requestedAlgoCount];
    checkBlasErrors(cublasLtMatmulAlgoGetHeuristic(
        handle, desc, layoutA, layoutB, layoutC, layoutD, preference,
        requestedAlgoCount, heuristicResult, &returnedAlgoCounts));
    std::vector<cublasLtMatmulAlgo_t> algos;
    for (int i = 0; i < returnedAlgoCounts; i++) {
        algos.emplace_back(heuristicResult[i].algo);
    }
    return algos;
}

double evalCublasLtKernel(const cublasLtHandle_t &handle,
                          const cublasLtMatmulDesc_t &desc, float *ptrA,
                          const cublasLtMatrixLayout_t &layoutA, float *ptrB,
                          const cublasLtMatrixLayout_t &layoutB, float *ptrC,
                          const cublasLtMatrixLayout_t &layoutC, float *bias,
                          const cublasLtMatrixLayout_t &layoutBias,
                          cublasLtMatmulAlgo_t *algo, float *workspace,
                          const size_t workspaceSize) {
    float alpha = 1.0f, beta = 0.0f;
    cudaEvent_t st, ed;
    cudaEventCreate(&st);
    cudaEventCreate(&ed);
    if (bias == nullptr) {
        for (int i = 0; i < numWarmup; i++) {
            checkBlasErrors(cublasLtMatmul(
                handle, desc, &alpha, ptrA, layoutA, ptrB, layoutB, &beta, ptrC,
                layoutC, ptrC, layoutC, algo, workspace, workspaceSize, 0));
        }
        cudaEventRecord(st, 0);
        for (int i = 0; i < numEval; i++) {
            checkBlasErrors(cublasLtMatmul(
                handle, desc, &alpha, ptrA, layoutA, ptrB, layoutB, &beta, ptrC,
                layoutC, ptrC, layoutC, algo, workspace, workspaceSize, 0));
        }
    } else {
        for (int i = 0; i < numWarmup; i++) {
            checkBlasErrors(cublasLtMatmul(
                handle, desc, &alpha, ptrA, layoutA, ptrB, layoutB, &beta, bias,
                layoutBias, ptrC, layoutC, algo, workspace, workspaceSize, 0));
        }
        cudaEventRecord(st, 0);
        for (int i = 0; i < numEval; i++) {
            checkBlasErrors(cublasLtMatmul(
                handle, desc, &alpha, ptrA, layoutA, ptrB, layoutB, &beta, bias,
                layoutBias, ptrC, layoutC, algo, workspace, workspaceSize, 0));
        }
    }
    cudaEventRecord(ed, 0);
    cudaEventSynchronize(st);
    cudaEventSynchronize(ed);
    float duration;
    cudaEventElapsedTime(&duration, st, ed);
    double time = duration / float(numEval);
    return time;
}

std::string getTag(const cublasLtMatmulAlgo_t &algo) {
    std::ostringstream sout;
    sout << "CUBLASLT";
    for (int i = 0; i < 8; i++) {
        sout << " 0x" << std::setfill('0') << std::setw(16) << std::hex
             << algo.data[i];
    }
    return sout.str();
}

bool verify(const cublasLtHandle_t &handle, int ba, int bb, int bc, int m, int n, int k,
            int rba, int rma, int rka,
            int rbb, int rkb, int rnb,
            int biasType, cublasLtEpilogue_t epilogue, int layoutIdA, int layoutIdB,
            int layoutIdC, size_t wsSize, const std::string &tag) {
    auto algo = MCCompiler::cublas_utils::getAlgo(tag);
    float *ptrA = nullptr, *ptrB = nullptr, *ptrC = nullptr, *bias = nullptr,
          *workspace = nullptr;
    int b = (ba < bb) ? bb : ba;
    checkCudaErrors(cudaMalloc((void **)&ptrA, rba * rma * rka * sizeof(float)));
    checkCudaErrors(cudaMalloc((void **)&ptrB, rbb * rkb * rnb * sizeof(float)));
    checkCudaErrors(cudaMalloc((void **)&ptrC, b * m * n * sizeof(float)));
    checkCudaErrors(cudaMalloc((void **)&workspace, wsSize * sizeof(float)));

    auto desc = MCCompiler::cublas_utils::getDesc(epilogue);
    auto layoutA = MCCompiler::cublas_utils::getLayout(b, m, k, rba, rma, rka, layoutIdA, ba);
    auto layoutB = MCCompiler::cublas_utils::getLayout(b, k, n, rbb, rkb, rnb, layoutIdB, bb);
    auto layoutC = MCCompiler::cublas_utils::getLayout(b, m, n, b, m, n, layoutIdC, b);

    curandGenerator_t gen;
    curandSafeCall(curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT));
    curandSafeCall(curandGenerateUniform(gen, ptrA, rba * rma * rka));
    curandSafeCall(curandGenerateUniform(gen, ptrB, rbb * rkb * rnb));
    // checkCudaErrors(cudaMemset(ptrC, 0, m * n * sizeof(float)));

    float *hostA = (float *)malloc(rba * rma * rka * sizeof(float));
    float *hostB = (float *)malloc(rbb * rkb * rnb * sizeof(float));
    float *hostC = (float *)malloc(b * m * n * sizeof(float));
    float *resC = (float *)malloc(b * m * n * sizeof(float));

    checkCudaErrors(cudaMemcpy(hostA, ptrA, rba * rma * rka * sizeof(float),
                            cudaMemcpyDeviceToHost));
    checkCudaErrors(cudaMemcpy(hostB, ptrB, rbb * rkb * rnb * sizeof(float),
                            cudaMemcpyDeviceToHost));

    std::vector<std::vector<int>> strideA = {
        {rma * rka, rka, 1}, {rka, rba * rka, 1}, {rma * rka, 1, rma}, {rma, 1, rba * rma}};
    std::vector<std::vector<int>> strideB = {
        {rkb * rnb, rnb, 1}, {rnb, rbb * rnb, 1}, {rkb * rnb, 1, rkb}, {rkb, 1, rbb * rkb}};
    std::vector<std::vector<int>> strideC = {
        {m * n, n, 1}, {n, b * n, 1}, {m * n, 1, m}, {m, 1, b * m}};
    
    if (ba == 1) { strideA[0][0] = strideA[1][0] = strideA[2][0] = strideA[3][0] = 0; }
    if (bb == 1) { strideB[0][0] = strideB[1][0] = strideB[2][0] = strideB[3][0] = 0; }

    for (int ib = 0; ib < b; ib++) {
        for (int im = 0; im < m; im++) {
            for (int in = 0; in < n; in++) {
                hostC[ib * strideC[layoutIdC][0] + im * strideC[layoutIdC][1] +
                      in * strideC[layoutIdC][2]] = 0;
                for (int ik = 0; ik < k; ik++) {
                    hostC[ib * strideC[layoutIdC][0] +
                          im * strideC[layoutIdC][1] +
                          in * strideC[layoutIdC][2]] +=
                        hostA[ib * strideA[layoutIdA][0] +
                              im * strideA[layoutIdA][1] +
                              ik * strideA[layoutIdA][2]] *
                        hostB[ib * strideB[layoutIdB][0] +
                              ik * strideB[layoutIdB][1] +
                              in * strideB[layoutIdB][2]];
                }
            }
        }
    }

    if (biasType == 0) {
        float alpha = 1.0f, beta = 0.0f;
        checkBlasErrors(cublasLtMatmul(handle, desc, &alpha, ptrA, layoutA, ptrB,
                                      layoutB, &beta, ptrC, layoutC, ptrC,
                                      layoutC, &algo, workspace, wsSize, 0));
    } else {
        float alpha = 1.0f, beta = 1.0f;
        checkCudaErrors(cudaMalloc((void **)&bias, bc * n * sizeof(float)));
        auto layoutBias = MCCompiler::cublas_utils::getLayoutBias(b, m, n, layoutIdC, bc);
        curandSafeCall(curandGenerateUniform(gen, bias, n * bc));
        float *hostBias = (float *)malloc(n * bc * sizeof(float));
        checkCudaErrors(cudaMemcpy(hostBias, bias, n * bc * sizeof(float),
                                cudaMemcpyDeviceToHost));
        checkBlasErrors(cublasLtMatmul(handle, desc, &alpha, ptrA, layoutA, ptrB,
                                      layoutB, &beta, bias, layoutBias, ptrC,
                                      layoutC, &algo, workspace, wsSize, 0));
        int stride_bias = bc == 1 ? 0 : n;
        for (int ib = 0; ib < b; ib++) {
            for (int im = 0; im < m; im++) {
                for (int in = 0; in < n; in++) {
                    hostC[ib * strideC[layoutIdC][0] +
                          im * strideC[layoutIdC][1] +
                          in * strideC[layoutIdC][2]] += hostBias[ib * stride_bias + in];
                }
            }
        }
    }
    checkCudaErrors(cudaMemcpy(resC, ptrC, b * m * n * sizeof(float),
                            cudaMemcpyDeviceToHost));

    for (int i = 0; i < b * m * n; i++) {
        if (std::abs(hostC[i] - resC[i]) / std::abs(hostC[i]) > 1e-4) {
            std::cout << i << " " << hostC[i] << " " << resC[i] << std::endl;
            return false;
        }
    }
    return true;
}

void evalCublasLt(int ba, int bb, int bc, int m, int n, int k,
                  int rba, int rma, int rka,
                  int rbb, int rkb, int rnb,
                  int biasType, int epilogue,
                  int layoutIdA, int layoutIdB,
                  int layoutIdC, size_t wsSize) {
    printf("evaluating %d %d %d %d %d %d with biastype %d epilogue %d layout %d "
           "%d %d wsSize %d\n",
           ba, bb, bc, m, n, k, biasType, epilogue, layoutIdA, layoutIdB,
           layoutIdC, wsSize);
    float *ptrA = nullptr, *ptrB = nullptr, *ptrC = nullptr, *bias = nullptr,
          *workspace = nullptr;
    int b = (ba < bb) ? bb : ba;
    checkCudaErrors(cudaMalloc((void **)&ptrA, rba * rma * rka * sizeof(float)));
    checkCudaErrors(cudaMalloc((void **)&ptrB, rbb * rkb * rnb * sizeof(float)));
    checkCudaErrors(cudaMalloc((void **)&ptrC, b * m * n * sizeof(float)));
    checkCudaErrors(cudaMalloc((void **)&workspace, wsSize * sizeof(float)));
    if (biasType != 0) {
        assert(bc == b || bc == 1);
        checkCudaErrors(cudaMalloc((void **)&bias, bc * n * sizeof(float)));
    }

    cublasLtHandle_t handle;
    cublasLtCreate(&handle);

    auto desc = MCCompiler::cublas_utils::getDesc((cublasLtEpilogue_t) epilogue);
    auto layoutA = MCCompiler::cublas_utils::getLayout(b, m, k, rba, rma, rka, layoutIdA, ba);
    auto layoutB = MCCompiler::cublas_utils::getLayout(b, k, n, rbb, rkb, rnb, layoutIdB, bb);
    auto layoutC = MCCompiler::cublas_utils::getLayout(b, m, n, b, m, n, layoutIdC, b);
    auto layoutBias = MCCompiler::cublas_utils::getLayoutBias(b, m, n, layoutIdC, bc);

    auto algos = getAlgo(handle, desc, layoutA, layoutB,
                         (biasType) ? layoutBias : layoutC, layoutC, wsSize);
    double bestTime = INFINITY;
    int bestIdx = -1;

    for (size_t i = 0; i < algos.size(); i++) {
        auto time = evalCublasLtKernel(handle, desc, ptrA, layoutA, ptrB,
                                       layoutB, ptrC, layoutC, bias, layoutBias,
                                       &algos[i], workspace, wsSize);
        if (time < bestTime) {
            bestTime = time;
            bestIdx = i;
        }
    }
    assert(bestIdx != -1);
    assert(!enableVerification ||
           verify(handle, ba, bb, bc, m, n, k, rba, rma, rka, rbb, rkb, rnb, biasType, (cublasLtEpilogue_t) epilogue, layoutIdA,
                  layoutIdB, layoutIdC, wsSize, getTag(algos[bestIdx])));

    checkBlasErrors(cublasLtMatrixLayoutDestroy(layoutA));
    checkBlasErrors(cublasLtMatrixLayoutDestroy(layoutB));
    checkBlasErrors(cublasLtMatrixLayoutDestroy(layoutC));
    checkBlasErrors(cublasLtMatmulDescDestroy(desc));
    checkBlasErrors(cublasLtDestroy(handle));
    checkCudaErrors(cudaFree(ptrA));
    checkCudaErrors(cudaFree(ptrB));
    checkCudaErrors(cudaFree(ptrC));
    checkCudaErrors(cudaFree(workspace));
    if (biasType != 0) {
        checkCudaErrors(cudaFree(bias));
    }
    printf("best time: %f\n", bestTime);
    printf("tag: %s\n", getTag(algos[bestIdx]).c_str());
}


} // namespace MCCompiler

int main(int argc, char* argv[]) {
    assert(argc  == 19);
    int ba = atoi(argv[1]);
    int bb = atoi(argv[2]);
    int bc = atoi(argv[3]);
    int m = atoi(argv[4]);
    int n = atoi(argv[5]);
    int k = atoi(argv[6]);
    int rba = atoi(argv[7]);
    int rma = atoi(argv[8]);
    int rka = atoi(argv[9]);
    int rbb = atoi(argv[10]);
    int rkb = atoi(argv[11]);
    int rnb = atoi(argv[12]);
    int biasType = atoi(argv[13]);
    int epilogue = atoi(argv[14]);
    int layoutIdA = atoi(argv[15]);
    int layoutIdB = atoi(argv[16]);
    int layoutIdC = atoi(argv[17]); 
    size_t wsSize = atoi(argv[18]);
    MCCompiler::evalCublasLt(ba, bb, bc, m, n, k, rba, rma, rka, rbb, rkb, rnb, biasType, epilogue, layoutIdA, layoutIdB, layoutIdC, wsSize);
}
// g++ mc/operators/cuda/cublas.cpp -o build/cublas_util -lcublas -lcudart -lcublasLt -lcurand 

// 10 1 1 1 3072 768 10 1 768 1 768 3072 1 1 0 0 0 1024
// linear [10, 1 768] [768, 3072] [3072] -> [10, 1, 3072]

// 12 12 12  10 10 64  36 10 64  36 64 10  1 1 3 1 0 1024
// 12 12 12  10 10 64  36 10 64  36 64 10  0 1 3 1 0 1024