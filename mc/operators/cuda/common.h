#pragma once
#define UNREACHABLE() { \
    printf("file %s line %i: unreachable!\n", __FILE__, __LINE__); \
    fflush(stdout); \
    exit(1); \
}

static const char *cublasGetErrorString(cublasStatus_t error) {
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
        default:
            return "<unknown>";
    }
    UNREACHABLE()
}

#define checkBlasErrors(stmt) { \
    cublasStatus_t err = stmt; \
    if (err != CUBLAS_STATUS_SUCCESS) {                          \
    fprintf(stderr, "%s in file %s, function %s, line %i: %04d %s\n", #stmt, __FILE__, __FUNCTION__, __LINE__, err, cublasGetErrorString(err)); \
    exit(1); \
    } \
}


#define checkCudaErrors(stmt) {                                 \
    cudaError_t err = stmt;                            \
    if (err != cudaSuccess) {                          \
    fprintf(stderr, "%s in file %s, function %s, line %i: %04d %s\n", #stmt, __FILE__, __FUNCTION__, __LINE__, err, cudaGetErrorString(err)); \
    exit(1); \
    }                                                  \
}
