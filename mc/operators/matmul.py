from mc.node import Node, IndexNode, gen_name
from typing import List, Optional, Tuple, Dict
from mc.types import TensorType
import onnx
import numpy as np
import enum
import copy
from mc.utils import CodeWriter
import os
import subprocess

# https://docs.nvidia.com/cuda/cublas/index.html#cublasltepilogue-t
class Epilogue(enum.Enum):
    CUBLASLT_EPILOGUE_DEFAULT = 1
    CUBLASLT_EPILOGUE_RELU = 2
    CUBLASLT_EPILOGUE_RELU_AUX = CUBLASLT_EPILOGUE_RELU | 128
    CUBLASLT_EPILOGUE_BIAS = 4
    CUBLASLT_EPILOGUE_RELU_BIAS = CUBLASLT_EPILOGUE_RELU | CUBLASLT_EPILOGUE_BIAS
    CUBLASLT_EPILOGUE_RELU_AUX_BIAS = CUBLASLT_EPILOGUE_RELU_AUX | CUBLASLT_EPILOGUE_BIAS
    CUBLASLT_EPILOGUE_DRELU = 8 | 128
    CUBLASLT_EPILOGUE_DRELU_BGRAD = CUBLASLT_EPILOGUE_DRELU | 16
    CUBLASLT_EPILOGUE_GELU = 32
    CUBLASLT_EPILOGUE_GELU_AUX = CUBLASLT_EPILOGUE_GELU | 128
    CUBLASLT_EPILOGUE_GELU_BIAS = CUBLASLT_EPILOGUE_GELU | CUBLASLT_EPILOGUE_BIAS
    CUBLASLT_EPILOGUE_GELU_AUX_BIAS = CUBLASLT_EPILOGUE_GELU_AUX | CUBLASLT_EPILOGUE_BIAS
    CUBLASLT_EPILOGUE_DGELU = 64 | 128
    CUBLASLT_EPILOGUE_DGELU_BGRAD = CUBLASLT_EPILOGUE_DGELU | 16
    CUBLASLT_EPILOGUE_BGRADA = 256
    CUBLASLT_EPILOGUE_BGRADB = 512

class UniMatMul(Node):
    # [size_b, size_m, size_k] * [size_b, size_k, size_n] -> [size_b, size_m, size_n]
    size_b: int
    size_m: int
    size_n: int
    size_k: int
    input_stride: List[Tuple[int, int, int]] # with length 2
    bias_stride: Tuple[int, int, int]
    output_stride: Tuple[int, int, int]
    input_offset: List[int]
    output_offset: int
    alpha: float
    beta: float
    epilogue: Epilogue
    matmul_tag: Optional[str]
    matmul_interface: Dict[str, str]

    def __init__(
            self, name: str,
            input_nodes: List[IndexNode], output_nodes: List[List[IndexNode]],
            input_types: List[TensorType], output_types: List[TensorType],
            input_constants: List[Optional[np.ndarray]],
            size_b: int = -1, size_m: int = -1, size_n: int = -1, size_k: int = -1,
            input0_stride: Tuple[int, int, int] = (0, 0, 0), input1_stride: Tuple[int, int, int] = (0, 0, 0),
            output_stride: Tuple[int, int, int] = (0, 0, 0),
            bias_stride: Optional[Tuple[int, int, int]]=None,
            alpha: float = 1.0, beta: float = 0.0,
            epilogue: Epilogue = Epilogue.CUBLASLT_EPILOGUE_DEFAULT,
            input_offset: List[int] = [0, 0, 0], output_offset: int = 0,
        ) -> None:
        super().__init__(name, input_nodes, output_nodes, input_types, output_types, input_constants)
        self.size_b = size_b
        self.size_m = size_m
        self.size_n = size_n
        self.size_k = size_k
        self.input_stride = [input0_stride, input1_stride]
        self.output_stride = output_stride
        self.bias_stride = bias_stride
        assert isinstance(input_offset, list)
        self.alpha = alpha
        self.beta = beta
        self.epilogue = epilogue
        self.input_offset = copy.deepcopy(input_offset)
        self.output_offset = output_offset
        self.matmul_interface = {}
        self.matmul_tag = None


    @classmethod
    def copy_attr_from(cls, node: 'UniMatMul', num_inputs = -1) -> 'UniMatMul':
        if num_inputs == -1:
            num_inputs = len(node.input_nodes)

        return cls(
            gen_name('UniMatMul'), [None] * num_inputs, [[]], [None] * num_inputs, [None], [],
            node.size_b, node.size_m, node.size_n, node.size_k,
            node.input_stride[0], node.input_stride[1], node.output_stride, node.bias_stride,
            node.alpha, node.beta, node.epilogue
        )

    def gen_matmul_tag(self):
        cache = {
            '10,1,1,3072,768,1,1,0,2,0,1024': 'CUBLASLT 0x000000000000000d 0x0000000100000000 0x0000000000000000 0x00002db70000000c 0x0000000000000000 0x000000000000004d 0x0000000000000050 0x0000000000000000',
            '10,1,1,3072,768,1,1,0,0,0,1024': 'CUBLASLT 0x000000000000000d 0x0000000100000000 0x0000000000000000 0x00002db700000002 0x0000000000000000 0x000000000000004d 0x0000000000000050 0x0000000000000000'
        }
        print(f"size_b={self.size_b}, size_m={self.size_m}, size_n={self.size_n}, size_k={self.size_k}")
        print(f"input_stride={self.input_stride}")
        print(f"output_stride={self.output_stride}")
        print(f"bias_stride={self.bias_stride}")
        print(f"alpha={self.alpha}, beta={self.beta}")
        print(f"epilogue={self.epilogue.name}")
        print(f"input_offset={self.input_offset}")
        print(f"output_offset={self.output_offset}")


        ba = self.size_b if self.input_stride[0][0] > 0 else 1
        bb = self.size_b if self.input_stride[1][0] > 0 else 1
        m = self.size_m
        n = self.size_n
        k = self.size_k
        biasType = 1 if len(self.input_types) == 3 else 0
        epilogue = self.epilogue
        layoutIdA = 0
        layoutIdB = 0
        layoutIdC = 0
        wsSize = 1024

        self.matmul_interface = {
            'ba': ba,
            'bb': bb,
            'm': m,
            'n': n,
            'k': k,
            'biasType': biasType,
            'epilogue': epilogue.value,
            'layoutIdA': layoutIdA,
            'layoutIdB': layoutIdB,
            'layoutIdC': layoutIdC,
            'wsSize': wsSize,
        }
        key = ",".join([str(x) for x in [ba, bb, m, n, k, biasType, epilogue.value, layoutIdA, layoutIdB, layoutIdC, wsSize]])
        print('key', key)
        if key in cache:
            self.matmul_tag = cache[key]
            return

        cublas_util_path = os.path.normpath(os.path.join(os.path.dirname(os.path.realpath(__file__)), '../../build/cublas_util'))
        cmd = f'{cublas_util_path} {ba} {bb} {m} {n} {k} {biasType} {epilogue.value} {layoutIdA} {layoutIdB} {layoutIdC} {wsSize}'
        proc = subprocess.Popen([cmd], stdout=subprocess.PIPE, shell=True)
        (out, err) = proc.communicate()
        if err is not None:
            raise RuntimeError(f"Failed to generate matmul tag with {cmd}: {err}")
        out = out.decode('utf-8')
        for line in out.splitlines():
            if line.startswith('tag: '):
                self.matmul_tag = line[5:]
                print(self.matmul_tag)
                return
        raise ValueError(f"Failed to parse tag from {cmd}: {out}")

    def get_global_params(self, node_name) -> List[Tuple[str, str]]:
        if self.matmul_tag is None:
            self.gen_matmul_tag()
        global_params = [
            ('cublasLtHandle_t', f'{node_name}__handle'),
            ('cublasLtMatmulDesc_t', f'{node_name}__desc'),
            ('cublasLtMatrixLayout_t', f'{node_name}__layoutA'),
            ('cublasLtMatrixLayout_t', f'{node_name}__layoutB'),
            ('cublasLtMatrixLayout_t', f'{node_name}__layoutC'),
            ('char*', f'{node_name}__workspace'),
            ('float', f'{node_name}__alpha'),
            ('float', f'{node_name}__beta'),
            ('cublasLtMatmulAlgo_t', f'{node_name}__algo'),
            ('size_t', f'{node_name}__wsSize')
        ]
        if self.matmul_interface['biasType'] == 1:
            global_params.append(('cublasLtMatrixLayout_t', f'{node_name}__layoutBias'))
        return global_params

    def get_init_code(self, node_name) -> str:
        if self.matmul_tag is None:
            self.gen_matmul_tag()
        ba, bb, m, n, k, biasType, epilogue, layoutIdA, layoutIdB, layoutIdC, wsSize = \
            self.matmul_interface['ba'], self.matmul_interface['bb'], self.matmul_interface['m'], \
            self.matmul_interface['n'], self.matmul_interface['k'], self.matmul_interface['biasType'], \
            self.matmul_interface['epilogue'], self.matmul_interface['layoutIdA'], \
            self.matmul_interface['layoutIdB'], self.matmul_interface['layoutIdC'], self.matmul_interface['wsSize']
        b = max(ba, bb)
        writer = CodeWriter()
        if self.beta != 0.0: raise NotImplementedError
        if biasType == 1: beta = 1.0
        else: beta = 0.0
        writer.write(f''' // {node_name}
checkBlasErrors(cublasLtCreate(&{node_name}__handle));
{node_name}__desc = MCCompiler::cublas_utils::getDesc( 
    (cublasLtEpilogue_t){epilogue});
{node_name}__layoutA = MCCompiler::cublas_utils::getLayout({b}, {m}, {k}, {layoutIdA}, {ba});
{node_name}__layoutB = MCCompiler::cublas_utils::getLayout({b}, {k}, {n}, {layoutIdB}, {bb});
{node_name}__layoutC = MCCompiler::cublas_utils::getLayout({b}, {m}, {n}, {layoutIdC}, {b});
checkCudaErrors(cudaMalloc((void **)&{node_name}__workspace, {wsSize}));
checkCudaErrors(cudaMalloc((void **)&{node_name}__alpha, sizeof(float)));
checkCudaErrors(cudaMalloc((void **)&{node_name}__beta, sizeof(float)));
{node_name}__alpha = {self.alpha}, {node_name}__beta = {beta};
{node_name}__algo = MCCompiler::cublas_utils::getAlgo("{self.matmul_tag}");
{node_name}__wsSize = {wsSize};
''')    
        if biasType == 1:
            writer.wl(f'{node_name}__layoutBias = MCCompiler::cublas_utils::getLayoutBias({b}, {m}, {n}, {layoutIdC}, {b});')
        return writer.get_code()

    def get_cuda_code(self, func_sig, node_name) -> str:
        if self.matmul_tag is None:
            self.gen_matmul_tag()
        writer = CodeWriter()
        writer.wl(func_sig)
        writer.block_start()
        if self.matmul_interface['biasType'] == 0:
            writer.wl(f'checkBlasErrors(cublasLtMatmul({node_name}__handle, {node_name}__desc, &{node_name}__alpha, input0, {node_name}__layoutA, input1, {node_name}__layoutB, &{node_name}__beta, ouptut0, {node_name}__layoutC, output0, {node_name}__layoutC, &{node_name}__algo, {node_name}__workspace, {node_name}__wsSize, 0));')
        else:
            writer.wl(f'checkBlasErrors(cublasLtMatmul({node_name}__handle, {node_name}__desc, &{node_name}__alpha, input0, {node_name}__layoutA, input1, {node_name}__layoutB, &{node_name}__beta, input2, {node_name}__layoutBias, output0, {node_name}__layoutC, &{node_name}__algo, {node_name}__workspace, {node_name}__wsSize, 0));')
        writer.block_end()
        return writer.get_code() 

    def __str__(self) -> str:
        s = super().__str__()
        attrs = []
        attrs.append('input_stride=' + str(self.input_stride))
        if self.epilogue != Epilogue.CUBLASLT_EPILOGUE_DEFAULT:
            attrs.append(f'epilogue={self.epilogue.name}')
        if self.alpha != 1.0:
            attrs.append(f'alpha={self.alpha}')
        for i in range(len(self.input_offset)):
            if self.input_offset[i] != 0:
                attrs.append(f'input_offset[{i}]={self.input_offset[i]}')
        if len(attrs) > 0:
            s = s + ' (' + ', '.join(attrs) + ')'
        return s


class UniMatMulNoBias(UniMatMul):
    def __init__(self, name: str,
                 input_nodes: List[IndexNode], output_nodes: List[List[IndexNode]],
                 input_types: List[TensorType], output_types: List[TensorType],
                 input_constants: List[Optional[np.ndarray]],
                 size_b: int = -1, size_m: int = -1, size_n: int = -1, size_k: int = -1,
                 input0_stride: Tuple[int, int, int] = (0, 0, 0), input1_stride: Tuple[int, int, int] = (0, 0, 0), output_stride: Tuple[int, int, int] = (0, 0, 0),
                 alpha: float = 1.0, beta: float = 0.0, epilogue: Epilogue = Epilogue.CUBLASLT_EPILOGUE_DEFAULT,
                 input_offset: List[int] = [0, 0], output_offset: int = 0,) -> None:
        super().__init__(name, input_nodes, output_nodes, input_types, output_types, input_constants, size_b, size_m, size_n, size_k, input0_stride, input1_stride, output_stride, None, alpha, beta, epilogue, input_offset, output_offset)



class MatMul(UniMatMulNoBias):
    def __init__(
        self, name:str,
        input_nodes:List['IndexNode'], output_nodes:List[List['IndexNode']],
        input_types:List[TensorType], output_types:List[TensorType],
        input_constants: List[Optional[np.ndarray]],
    ) -> None:
        # super().__init__(name, input_nodes, output_nodes, input_types, output_types)
        if len(input_types[0].shape) <= 2 and len(input_types[1].shape) <= 2:
            size_b = 1
        elif len(input_types[0].shape) == 3 and len(input_types[1].shape) == 3:
            size_b = max(input_types[0].shape[0], input_types[1].shape[0])
        elif len(input_types[0].shape) == 3:
            size_b = input_types[0].shape[0]
        elif len(input_types[1].shape) == 3:
            size_b = input_types[1].shape[0]
        elif len(input_types[0].shape) > 3 or len(input_types[1].shape) > 3:
            raise NotImplementedError
        else:
            raise ValueError("unreachable")
        size_m = input_types[0].shape[-2]
        size_k = input_types[0].shape[-1]
        size_n = input_types[1].shape[-1]
        input0_stride = [0, size_k, 1]
        if len(input_types[0].shape) == 3:
            input0_stride[0] = size_m * size_k
        input1_stride = [0, size_n, 1]
        if len(input_types[1].shape) == 3:
            input1_stride[0] = size_n * size_k
        output_stride = [size_m * size_k, size_k, 1]
        super().__init__(name, input_nodes, output_nodes, input_types, output_types, input_constants, size_b, size_m, size_n, size_k, input0_stride, input1_stride, output_stride)


# [m, k] * [k, n] -> [m, n]
class Gemm(UniMatMul):
    @classmethod
    def from_onnx(
        cls, name:str,
        input_nodes:List['IndexNode'], output_nodes:List[List['IndexNode']],
        input_types:List[TensorType], output_types:List[TensorType],
        input_constants: List[Optional[np.ndarray]],
        onnx_node:onnx.NodeProto,
    ) -> None:
        size_b = 1
        attributes = {attr.name: attr for attr in onnx_node.attribute}
        alpha = attributes["alpha"].f
        beta = attributes["beta"].f
        transA = attributes["transA"].i if "transA" in attributes else 0
        transB = attributes["transB"].i if "transB" in attributes else 0
        if transA:
            size_k, size_m = input_types[0].shape
            input0_stride = [0, 1, size_m]
        else:
            size_m, size_k = input_types[0].shape
            input0_stride = [0, size_k, 1]
        if transB:
            size_n = input_types[1].shape[0]
            input1_stride = [0, 1, size_k]
        else:
            size_n = input_types[1].shape[1]
            input1_stride = [0, size_n, 1]
        bias_stride = [0, 0, 1]
        output_stride = [size_m * size_n, size_n, 1]
        return cls(name, input_nodes, output_nodes, input_types, output_types, input_constants, size_b, size_m, size_n, size_k, input0_stride, input1_stride, output_stride, bias_stride, alpha=alpha, beta=beta)
