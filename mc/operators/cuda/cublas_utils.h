#include <cublas_v2.h>
#include <cublasLt.h>
#include <sstream>

namespace MCCompiler {
namespace cublas_utils {

cublasLtMatmulAlgo_t getAlgo(const std::string &tag) {
    cublasLtMatmulAlgo_t algo;
    std::istringstream sin(tag);
    std::string head;
    sin >> head;
    assert(head == "CUBLASLT");
    for (int i = 0; i < 8; i++) {
        sin >> std::hex >> algo.data[i];
    }
    return algo;
}

cublasLtMatmulDesc_t getDesc(cublasLtEpilogue_t epilogue) {
    auto transa = CUBLAS_OP_N;
    auto transb = CUBLAS_OP_N;
    cublasLtMatmulDesc_t desc;
    checkBlasErrors(cublasLtMatmulDescCreate(&desc, CUBLAS_COMPUTE_32F_FAST_TF32,
                                            CUDA_R_32F));
    checkBlasErrors(cublasLtMatmulDescSetAttribute(
        desc, CUBLASLT_MATMUL_DESC_TRANSA, &transa, sizeof(transa)));
    checkBlasErrors(cublasLtMatmulDescSetAttribute(
        desc, CUBLASLT_MATMUL_DESC_TRANSB, &transb, sizeof(transb)));
    checkBlasErrors(cublasLtMatmulDescSetAttribute(
        desc, CUBLASLT_MATMUL_DESC_EPILOGUE, &epilogue, sizeof(epilogue)));
    return desc;
}


cublasLtMatrixLayout_t getLayout(int b, int m, int n, int layoutId, int bThis) {
    cublasLtMatrixLayout_t layout;

    size_t stride0 = (layoutId & 2) ? m : n;
    size_t stride1;
    if (layoutId & 1) {
        stride1 = stride0;
        stride0 *= b;
    } else {
        stride1 = m * n;
    }
    if (bThis == 1) {
        stride1 = 0;
    }
    checkBlasErrors(
        cublasLtMatrixLayoutCreate(&layout, CUDA_R_32F, m, n, stride0));
    auto layoutOrder = (layoutId & 2) ? CUBLASLT_ORDER_COL : CUBLASLT_ORDER_ROW;
    checkBlasErrors(
        cublasLtMatrixLayoutSetAttribute(layout, CUBLASLT_MATRIX_LAYOUT_ORDER,
                                         &layoutOrder, sizeof(layoutOrder)));
    checkBlasErrors(cublasLtMatrixLayoutSetAttribute(
        layout, CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT, &b, sizeof(b)));
    checkBlasErrors(cublasLtMatrixLayoutSetAttribute(
        layout, CUBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET, &stride1,
        sizeof(stride1)));
    return layout;
}

cublasLtMatrixLayout_t getLayoutBias(int b, int m, int n, int layoutId,
                                     int bThis) {
    cublasLtMatrixLayout_t layout;
    size_t stride = 0;
    checkBlasErrors(cublasLtMatrixLayoutCreate(&layout, CUDA_R_32F, m, n, 0));
    auto layoutOrder = (layoutId & 2) ? CUBLASLT_ORDER_COL : CUBLASLT_ORDER_ROW;
    checkBlasErrors(
        cublasLtMatrixLayoutSetAttribute(layout, CUBLASLT_MATRIX_LAYOUT_ORDER,
                                         &layoutOrder, sizeof(layoutOrder)));
    checkBlasErrors(cublasLtMatrixLayoutSetAttribute(
        layout, CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT, &b, sizeof(b)));
    checkBlasErrors(cublasLtMatrixLayoutSetAttribute(
        layout, CUBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET, &stride,
        sizeof(stride)));
    return layout;
}

}
}