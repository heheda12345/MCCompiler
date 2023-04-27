#include <float.h>
#include <assert.h>
namespace MCCompiler {
namespace SoftMax {

__global__ void softmax_wrap_reduce(float* input0, float* output0, int row_size, int num_row) {
    int wrap_id = blockIdx.x * blockDim.x / 32 + (threadIdx.x >> 5);
    if (wrap_id >= num_row) return;
    int lane_id = threadIdx.x & 31;
    float local[32];
    int num_val = 0;
    for (int i = lane_id; i < row_size; i += 32, num_val++) {
        local[num_val] = input0[wrap_id * row_size + i];
    }
    float max_value = lane_id < row_size ? local[0]: -FLT_MAX;
    for (int i = 1; i < num_val; i++){
        max_value = max(max_value, local[i]);
    }
    max_value = max(max_value, __shfl_xor_sync(0xffffffff, max_value, 16));
    max_value = max(max_value, __shfl_xor_sync(0xffffffff, max_value, 8));
    max_value = max(max_value, __shfl_xor_sync(0xffffffff, max_value, 4));
    max_value = max(max_value, __shfl_xor_sync(0xffffffff, max_value, 2));
    max_value = max(max_value, __shfl_xor_sync(0xffffffff, max_value, 1));
    
    float sum = 0;
    for (int i = 0; i < num_val; i++) {
        local[i] = expf(local[i] - max_value);
        sum += local[i];
    }

    sum += __shfl_xor_sync(0xffffffff, sum, 16);
    sum += __shfl_xor_sync(0xffffffff, sum, 8);
    sum += __shfl_xor_sync(0xffffffff, sum, 4);
    sum += __shfl_xor_sync(0xffffffff, sum, 2);
    sum += __shfl_xor_sync(0xffffffff, sum, 1);

    for (int i = 0; i < num_val; i++) {
        output0[wrap_id * row_size + lane_id + i * 32] = local[i] / sum;
    }
}

void softmax_last_col(float* input0, float* output0, int row_size, int num_row) {
    assert(row_size <= 1024);
    softmax_wrap_reduce<<<(num_row + 3) / 4, 128>>>(input0, output0, row_size, num_row);
}


}
}