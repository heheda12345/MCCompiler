namespace MCCompiler {
namespace element_wise {
template<typename T>
__global__ void expand_kernel(T* input0, T* output0, int row_size, int num_rows) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= row_size * num_rows) return;
    output0[tid] = input0[tid % row_size];
}

template<typename T>
__global__ void add_kernel(T* input0, T* input1, T* output0, int n) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n) return;
    output0[tid] = input0[tid] + input1[tid];
}

template<typename T>
__global__ void where_kernel(bool* cond, T* true_branch, T* false_branch, T* output, int n) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n) return;
    output[tid] = cond[tid] ? true_branch[0] : false_branch[tid];
}

template<typename T>
void expand(T* input0, T* output0, int row_size, int num_rows) {
    expand_kernel<<<(row_size * num_rows + 127) / 128, 128>>>(input0, output0, row_size, num_rows);
}

template<typename T>
void add(T* input0, T* input1, T* output0, int n) {
    add_kernel<<<(n + 127) / 128, 128>>>(input0, input1, output0, n);
}

template<typename T>
void where(bool* cond, T* true_branch, T* false_branch, T* output, int n) {
    where_kernel<<<(n + 127) / 128, 128>>>(cond, true_branch, false_branch, output, n);
}

}
}