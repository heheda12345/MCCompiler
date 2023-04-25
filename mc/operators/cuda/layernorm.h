namespace MCCompiler {
namespace LayerNorm {

__global__ void layernorm_coL_768_kernel(float *ptr_0, float *ptr_8, float *ptr_10, float *ptr_11, int num_row, float eps) {
    int lane_id = threadIdx.x % 32;
    int group_id = threadIdx.x / 32;
    int parallel_idx = blockIdx.x * blockDim.x / 32 + group_id;
    float ptr_336[24];
    float ptr_338[12];
    float ptr_340[1];
    float ptr_394[24];
    float ptr_553[24];
    float ptr_623[24];
    float ptr_625[12];
    float ptr_627[1];
    float ptr_787[1];
    float ptr_823[1];
    float ptr_827[24];
    float ptr_994[24];
    float ptr_996[24];
    float ptr_1116[24];
    float ptr_1118[24];
    float ptr_1120[24];
    float ptr_390[1];
    float ptr_392[2];
    float ptr_825[2];
    float ptr_342[1];
    float ptr_629[1];
    for (int loop_idx = parallel_idx; loop_idx < num_row; loop_idx += 320) {
        int offset_1740865985 = loop_idx * 768;
        #pragma unroll
        for (int idx_num = 0; idx_num < 12; idx_num++) {
            (float2 &)ptr_336[(idx_num * 2)] = (float2 &)
            ptr_0[(((idx_num * 64) + (lane_id * 2)) + offset_1740865985)];
        }
        #pragma unroll
        for (int idx_num = 0; idx_num < 12; idx_num++) {
            (float2 &)ptr_996[(idx_num * 2)] =
            (float2 &)ptr_8[((idx_num * 64) + (lane_id * 2))];
        }
        #pragma unroll
        for (int idx_num = 0; idx_num < 12; idx_num++) {
            (float2 &)ptr_1118[(idx_num * 2)] =
            (float2 &)ptr_10[((idx_num * 64) + (lane_id * 2))];
        }
        #pragma unroll
        for (int idx_num = 0; idx_num < 12; idx_num++) {
            ptr_342[0] = ptr_336[(idx_num * 2)];
            #pragma unroll
            for (int offset = 16; offset > 0; offset /= 2) {
                ptr_342[0] += __shfl_down_sync(0xffffffff, ptr_342[0], offset);
            }
            if (lane_id == 0) {
                ptr_338[idx_num] = ptr_342[0];
            }
            ptr_342[0] = ptr_336[((idx_num * 2) + 1)];
            #pragma unroll
            for (int offset = 16; offset > 0; offset /= 2) {
                ptr_342[0] += __shfl_down_sync(0xffffffff, ptr_342[0], offset);
            }
            if (lane_id == 0) {
                ptr_338[idx_num] += ptr_342[0];
            }
            ptr_338[idx_num] /= 64;
        }
        ptr_340[0] = ptr_338[0];
        #pragma unroll
        for (int idx_num = 1; idx_num < 12; idx_num++) {
            ptr_340[0] += ptr_338[idx_num];
        }
        ptr_340[0] /= 12;
        if (lane_id < 1) {
            ptr_390[0] = ptr_340[0];
        }
        #pragma unroll
        for (int idx_num = 0; idx_num < 1; idx_num++) {
            ptr_392[0] = __shfl_sync(0xffffffff, ptr_390[0], 0);
            ptr_392[1] = ptr_392[0];
        }
        for (int idx_num = 0; idx_num < 12; idx_num++) {
            ptr_394[(idx_num * 2)] = ptr_392[0];
            ptr_394[((idx_num * 2) + 1)] = ptr_392[0];
        }
        #pragma unroll
        for (int idx_num = 0; idx_num < 12; idx_num++) {
            ptr_553[(idx_num * 2)] =
            ptr_336[(idx_num * 2)] - ptr_394[(idx_num * 2)];
            ptr_553[((idx_num * 2) + 1)] =
            ptr_336[((idx_num * 2) + 1)] - ptr_394[((idx_num * 2) + 1)];
        }
        #pragma unroll
        for (int idx_num = 0; idx_num < 12; idx_num++) {
            ptr_623[(idx_num * 2)] =
            ptr_553[(idx_num * 2)] * ptr_553[(idx_num * 2)];
            ptr_623[((idx_num * 2) + 1)] =
            ptr_553[((idx_num * 2) + 1)] * ptr_553[((idx_num * 2) + 1)];
        }
        #pragma unroll
        for (int idx_num = 0; idx_num < 12; idx_num++) {
            ptr_629[0] = ptr_623[(idx_num * 2)];
            #pragma unroll
            for (int offset = 16; offset > 0; offset /= 2) {
                ptr_629[0] += __shfl_down_sync(0xffffffff, ptr_629[0], offset);
            }
            if (lane_id == 0) {
                ptr_625[idx_num] = ptr_629[0];
            }
            ptr_629[0] = ptr_623[((idx_num * 2) + 1)];
            #pragma unroll
            for (int offset = 16; offset > 0; offset /= 2) {
                ptr_629[0] += __shfl_down_sync(0xffffffff, ptr_629[0], offset);
            }
            if (lane_id == 0) {
                ptr_625[idx_num] += ptr_629[0];
            }
            ptr_625[idx_num] /= 64;
        }
        ptr_627[0] = ptr_625[0];
        #pragma unroll
        for (int idx_num = 1; idx_num < 12; idx_num++) {
            ptr_627[0] += ptr_625[idx_num];
        }
        ptr_627[0] /= 12;
        if (lane_id < 1) {
            #pragma unroll
            for (int idx_num = 0; idx_num < 1; idx_num++) {
                ptr_787[0] = ptr_627[0] + eps;
            }
        }
        if (lane_id < 1) {
            #pragma unroll
            for (int idx_num = 0; idx_num < 1; idx_num++) {
                ptr_823[0] = sqrtf(ptr_787[0]);
            }
        }
        #pragma unroll
        for (int idx_num = 0; idx_num < 1; idx_num++) {
            ptr_825[0] = __shfl_sync(0xffffffff, ptr_823[0], 0);
            ptr_825[1] = ptr_825[0];
        }
        for (int idx_num = 0; idx_num < 12; idx_num++) {
            ptr_827[(idx_num * 2)] = ptr_825[0];
            ptr_827[((idx_num * 2) + 1)] = ptr_825[0];
        }
        #pragma unroll
        for (int idx_num = 0; idx_num < 12; idx_num++) {
            ptr_994[(idx_num * 2)] =
            ptr_553[(idx_num * 2)] / ptr_827[(idx_num * 2)];
            ptr_994[((idx_num * 2) + 1)] =
            ptr_553[((idx_num * 2) + 1)] / ptr_827[((idx_num * 2) + 1)];
        }
        #pragma unroll
        for (int idx_num = 0; idx_num < 12; idx_num++) {
            ptr_1116[(idx_num * 2)] =
            ptr_994[(idx_num * 2)] * ptr_996[(idx_num * 2)];
            ptr_1116[((idx_num * 2) + 1)] =
            ptr_994[((idx_num * 2) + 1)] * ptr_996[((idx_num * 2) + 1)];
        }
        #pragma unroll
        for (int idx_num = 0; idx_num < 12; idx_num++) {
            ptr_1120[(idx_num * 2)] =
            ptr_1116[(idx_num * 2)] + ptr_1118[(idx_num * 2)];
            ptr_1120[((idx_num * 2) + 1)] =
            ptr_1116[((idx_num * 2) + 1)] + ptr_1118[((idx_num * 2) + 1)];
        }
        #pragma unroll
        for (int idx_num = 0; idx_num < 12; idx_num++) {
            (float2 &)
            ptr_11[(((idx_num * 64) + (lane_id * 2)) + offset_1740865985)] =
            (float2 &)ptr_1120[(idx_num * 2)];
        }
    }
}

template <int reduce_size, typename T>
void layernorm(T* input, T* gamma, T* beta, T* output, int num_row, T eps);

template<> void layernorm<768, float>(float* input, float* gamma, float* beta, float* output, int num_row, float eps) {
    layernorm_coL_768_kernel<<<80, 128>>>(input, gamma, beta, output, num_row, eps);
}

}
}