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
constexpr bool enableVerification = false;

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

bool verify(const cublasLtHandle_t &handle, int ba, int bb, int m, int n, int k,
            int biasType, cublasLtEpilogue_t epilogue, int layoutIdA, int layoutIdB,
            int layoutIdC, size_t wsSize, const std::string &tag) {
    auto algo = MCCompiler::cublas_utils::getAlgo(tag);
    float *ptrA = nullptr, *ptrB = nullptr, *ptrC = nullptr, *bias = nullptr,
          *workspace = nullptr;
    int b = (ba < bb) ? bb : ba;
    checkCudaErrors(cudaMalloc((void **)&ptrA, ba * m * k * sizeof(float)));
    checkCudaErrors(cudaMalloc((void **)&ptrB, bb * k * n * sizeof(float)));
    checkCudaErrors(cudaMalloc((void **)&ptrC, b * m * n * sizeof(float)));
    checkCudaErrors(cudaMalloc((void **)&workspace, wsSize * sizeof(float)));

    auto desc = getDesc(epilogue);
    auto layoutA = getLayout(b, m, k, layoutIdA, ba);
    auto layoutB = getLayout(b, k, n, layoutIdB, bb);
    auto layoutC = getLayout(b, m, n, layoutIdC, b);

    curandGenerator_t gen;
    curandSafeCall(curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT));
    curandSafeCall(curandGenerateUniform(gen, ptrA, ba * m * k));
    curandSafeCall(curandGenerateUniform(gen, ptrB, bb * k * n));
    // checkCudaErrors(cudaMemset(ptrC, 0, m * n * sizeof(float)));

    float *hostA = (float *)malloc(ba * m * k * sizeof(float));
    float *hostB = (float *)malloc(bb * k * n * sizeof(float));
    float *hostC = (float *)malloc(b * m * n * sizeof(float));
    float *resC = (float *)malloc(b * m * n * sizeof(float));

    checkCudaErrors(cudaMemcpy(hostA, ptrA, ba * m * k * sizeof(float),
                            cudaMemcpyDeviceToHost));
    checkCudaErrors(cudaMemcpy(hostB, ptrB, bb * k * n * sizeof(float),
                            cudaMemcpyDeviceToHost));

    std::vector<std::vector<int>> strideA = {
        {m * k, k, 1}, {k, ba * k, 1}, {m * k, 1, m}, {m, 1, ba * m}};
    std::vector<std::vector<int>> strideB = {
        {k * n, n, 1}, {n, bb * n, 1}, {k * n, 1, k}, {k, 1, bb * k}};
    std::vector<std::vector<int>> strideC = {
        {m * n, n, 1}, {n, b * n, 1}, {m * n, 1, m}, {m, 1, b * m}};

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
        checkCudaErrors(cudaMalloc((void **)&bias, n * sizeof(float)));
        auto layoutBias = getLayoutBias(b, m, n, layoutIdC, b);
        curandSafeCall(curandGenerateUniform(gen, bias, n));
        float *hostBias = (float *)malloc(n * sizeof(float));
        checkCudaErrors(cudaMemcpy(hostBias, bias, n * sizeof(float),
                                cudaMemcpyDeviceToHost));
        checkBlasErrors(cublasLtMatmul(handle, desc, &alpha, ptrA, layoutA, ptrB,
                                      layoutB, &beta, bias, layoutBias, ptrC,
                                      layoutC, &algo, workspace, wsSize, 0));
        for (int ib = 0; ib < b; ib++) {
            for (int im = 0; im < m; im++) {
                for (int in = 0; in < n; in++) {
                    hostC[ib * strideC[layoutIdC][0] +
                          im * strideC[layoutIdC][1] +
                          in * strideC[layoutIdC][2]] += hostBias[in];
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

void evalCublasLt(int ba, int bb, int m, int n, int k,
                         int biasType, int epilogue,
                         int layoutIdA, int layoutIdB,
                         int layoutIdC, size_t wsSize) {
    printf("evaluating %d %d %d %d %d with biastype %d epilogue %d layout %d "
           "%d %d wsSize %d\n",
           ba, bb, m, n, k, biasType, epilogue, layoutIdA, layoutIdB,
           layoutIdC, wsSize);
    float *ptrA = nullptr, *ptrB = nullptr, *ptrC = nullptr, *bias = nullptr,
          *workspace = nullptr;
    int b = (ba < bb) ? bb : ba;
    checkCudaErrors(cudaMalloc((void **)&ptrA, ba * m * k * sizeof(float)));
    checkCudaErrors(cudaMalloc((void **)&ptrB, bb * k * n * sizeof(float)));
    checkCudaErrors(cudaMalloc((void **)&ptrC, b * m * n * sizeof(float)));
    checkCudaErrors(cudaMalloc((void **)&workspace, wsSize * sizeof(float)));
    if (biasType != 0) {
        checkCudaErrors(cudaMalloc((void **)&bias, n * sizeof(float)));
    }

    cublasLtHandle_t handle;
    cublasLtCreate(&handle);

    auto desc = getDesc((cublasLtEpilogue_t) epilogue);
    auto layoutA = getLayout(b, m, k, layoutIdA, ba);
    auto layoutB = getLayout(b, k, n, layoutIdB, bb);
    auto layoutC = getLayout(b, m, n, layoutIdC, b);
    auto layoutBias = getLayoutBias(b, m, n, layoutIdC, b);

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
           verify(handle, ba, bb, m, n, k, biasType, (cublasLtEpilogue_t) epilogue, layoutIdA,
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
    assert(argc  == 12);
    int ba = atoi(argv[1]);
    int bb = atoi(argv[2]);
    int m = atoi(argv[3]);
    int n = atoi(argv[4]);
    int k = atoi(argv[5]);
    int biasType = atoi(argv[6]);
    int epilogue = atoi(argv[7]);
    int layoutIdA = atoi(argv[8]);
    int layoutIdB = atoi(argv[9]);
    int layoutIdC = atoi(argv[10]); 
    size_t wsSize = atoi(argv[11]);
    MCCompiler::evalCublasLt(ba, bb, m, n, k, biasType, epilogue, layoutIdA, layoutIdB, layoutIdC, wsSize);
}
// g++ mc/operators/cuda/cublas.cpp -o build/cublas_util