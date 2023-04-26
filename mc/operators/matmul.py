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
    # [size_ba, size_m, size_k] * [size_bb, size_k, size_n] -> [size_b, size_m, size_n]
    size_ba: int
    size_bb: int
    size_m: int
    size_n: int
    size_k: int
    input_perm: List[Tuple[int, int, int]] # with length 2
    bias_perm: Tuple[int, int, int]
    output_perm: Tuple[int, int, int]
    input_offset: List[int]
    output_offset: int
    alpha: float
    beta: float
    epilogue: Epilogue
    matmul_tag: Optional[str]
    matmul_interface: Dict[str, str]
    real_input_shape: List[List[int]]
    # perm: matmul(input0.reshape().transpose(input0.perm), input1.reshape().transpose(input1.perm)).transpose(output0.perm).reshape()

    def __init__(
            self, name: str,
            input_nodes: List[IndexNode], output_nodes: List[List[IndexNode]],
            input_types: List[TensorType], output_types: List[TensorType],
            input_constants: List[Optional[np.ndarray]],
            size_ba: int = -1, size_bb: int = -1, size_m: int = -1, size_n: int = -1, size_k: int = -1,
            input0_perm: Tuple[int, int, int] = (0, 1, 2), input1_perm: Tuple[int, int, int] = (0, 0, 0),
            output_perm: Tuple[int, int, int] = (0, 0, 0),
            bias_perm: Optional[Tuple[int, int, int]]=None,
            alpha: float = 1.0, beta: float = 0.0,
            epilogue: Epilogue = Epilogue.CUBLASLT_EPILOGUE_DEFAULT,
            input_offset: List[int] = [0, 0, 0], output_offset: int = 0,
            real_input_shape: Optional[List[List[int]]] = None,
        ) -> None:
        super().__init__(name, input_nodes, output_nodes, input_types, output_types, input_constants)
        self.size_ba = size_ba
        self.size_bb = size_bb
        self.size_m = size_m
        self.size_n = size_n
        self.size_k = size_k
        self.input_perm = [input0_perm, input1_perm]
        self.output_perm = output_perm
        self.bias_perm = bias_perm
        assert isinstance(input_offset, list)
        self.alpha = alpha
        self.beta = beta
        self.epilogue = epilogue
        self.input_offset = copy.deepcopy(input_offset)
        self.output_offset = output_offset
        if real_input_shape is None:
            real_input_shape = [[self.size_ba, self.size_m, self.size_k], [self.size_bb, self.size_k, self.size_n]]
        self.real_input_shape = real_input_shape
        self.matmul_interface = {}
        self.matmul_tag = None

    @classmethod
    def copy_attr_from(cls, node: 'UniMatMul', num_inputs = -1) -> 'UniMatMul':
        if num_inputs == -1:
            num_inputs = len(node.input_nodes)

        return cls(
            gen_name('UniMatMul'), [None] * num_inputs, [[]], [None] * num_inputs, [None], [],
            node.size_ba, node.size_bb, node.size_m, node.size_n, node.size_k,
            node.input_perm[0], node.input_perm[1], node.output_perm, node.bias_perm,
            node.alpha, node.beta, node.epilogue
        )

    def get_layout_id_from_perm(self, perm: Tuple[int, int, int]) -> int:
        assert isinstance(perm, tuple)
        if perm == (0, 1, 2):
            return 0
        elif perm == (1, 0, 2):
            return 1
        elif perm == (0, 2, 1):
            return 2
        elif perm == (1, 2, 0):
            return 3
        else:
            raise ValueError(f"Unknown perm {perm}")

    def gen_matmul_tag(self):
        cache = {
            # "10,1,1,3072,768,1,1,0,0,0,1024": "CUBLASLT 0x000000000000000d 0x0000000100000000 0x0000000000000000 0x00002db700000002 0x0000000000000000 0x000000000000004d 0x0000000000000050 0x0000000000000000",
            # "10,1,1,2304,768,1,1,0,0,0,1024": "CUBLASLT 0x000000000000000d 0x0000000100000000 0x0000000000000000 0x00002db700000002 0x0000000000000000 0x000000000000004d 0x0000000000000050 0x0000000000000000"
        }
        print(self)
        print(f"input_shape={self.input_types}")
        print(f"output_shape={self.output_types}")
        print(f"size_ba={self.size_ba}, size_bb={self.size_bb}, size_m={self.size_m}, size_n={self.size_n}, size_k={self.size_k}")
        print(f"input_perm={self.input_perm}")
        print(f"output_perm={self.output_perm}")
        print(f"bias_perm={self.bias_perm}")
        print(f"alpha={self.alpha}, beta={self.beta}")
        print(f"epilogue={self.epilogue.name}")
        print(f"input_offset={self.input_offset}")
        print(f"output_offset={self.output_offset}")
        print(f"real_input_shape={self.real_input_shape}")

        ba = self.size_ba
        bb = self.size_bb
        if len(self.input_types) == 3:
            bc = self.input_types[2].size() // self.size_n
        else:
            bc = 1
        m = self.size_m
        n = self.size_n
        k = self.size_k
        biasType = 1 if len(self.input_types) == 3 else 0
        epilogue = self.epilogue
        
        layoutIdA = self.get_layout_id_from_perm(self.input_perm[0])
        layoutIdB = self.get_layout_id_from_perm(self.input_perm[1])
        layoutIdC = 0
        wsSize = 1024

        self.matmul_interface = {
            'ba': ba,
            'bb': bb,
            'bc': bc,
            'm': m,
            'n': n,
            'k': k,
            'rba': self.real_input_shape[0][0],
            'rma': self.real_input_shape[0][1],
            'rka': self.real_input_shape[0][2],
            'rbb': self.real_input_shape[1][0],
            'rkb': self.real_input_shape[1][1],
            'rnb': self.real_input_shape[1][2],
            'biasType': biasType,
            'epilogue': epilogue.value,
            'layoutIdA': layoutIdA,
            'layoutIdB': layoutIdB,
            'layoutIdC': layoutIdC,
            'wsSize': wsSize,
        }
        call_params = [str(x) for x in self.matmul_interface.values()]
        key = ",".join(call_params)
        print('key', key)
        if key in cache:
            self.matmul_tag = cache[key]
            return

        cublas_util_path = os.path.normpath(os.path.join(os.path.dirname(os.path.realpath(__file__)), '../../build/cublas_util'))
        cmd = f'{cublas_util_path} {" ".join(call_params)}'
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
        ba, bb, bc, m, n, k, rba, rma, rka, rbb, rkb, rnb, biasType, epilogue, layoutIdA, layoutIdB, layoutIdC, wsSize = self.matmul_interface.values()
        b = max(ba, bb)
        writer = CodeWriter()
        if self.beta != 0.0: raise NotImplementedError
        if biasType == 1: beta = 1.0
        else: beta = 0.0
        writer.write(f''' // {node_name}
checkBlasErrors(cublasLtCreate(&{node_name}__handle));
{node_name}__desc = MCCompiler::cublas_utils::getDesc( 
    (cublasLtEpilogue_t){epilogue});
{node_name}__layoutA = MCCompiler::cublas_utils::getLayout({b}, {m}, {k}, {rba}, {rma}, {rka},  {layoutIdA}, {ba});
{node_name}__layoutB = MCCompiler::cublas_utils::getLayout({b}, {k}, {n}, {rbb}, {rkb}, {rnb}, {layoutIdB}, {bb});
{node_name}__layoutC = MCCompiler::cublas_utils::getLayout({b}, {m}, {n}, {b}, {m}, {n}, {layoutIdC}, {b});
checkCudaErrors(cudaMalloc((void **)&{node_name}__workspace, {wsSize}));
checkCudaErrors(cudaMalloc((void **)&{node_name}__alpha, sizeof(float)));
checkCudaErrors(cudaMalloc((void **)&{node_name}__beta, sizeof(float)));
{node_name}__alpha = {self.alpha}, {node_name}__beta = {beta};
{node_name}__algo = MCCompiler::cublas_utils::getAlgo("{self.matmul_tag}");
{node_name}__wsSize = {wsSize};
''')    
        if biasType == 1:
            writer.wl(f'{node_name}__layoutBias = MCCompiler::cublas_utils::getLayoutBias({b}, {m}, {n}, {layoutIdC}, {bc});')
        return writer.get_code()

    def get_cuda_code(self, func_sig, node_name) -> str:
        if self.matmul_tag is None:
            self.gen_matmul_tag()
        writer = CodeWriter()
        writer.wl(func_sig)
        writer.block_start()
        if self.matmul_interface['biasType'] == 0:
            writer.wl(f'checkBlasErrors(cublasLtMatmul({node_name}__handle, {node_name}__desc, &{node_name}__alpha, input0 + {self.input_offset[0]}, {node_name}__layoutA, input1 + {self.input_offset[1]}, {node_name}__layoutB, &{node_name}__beta, output0, {node_name}__layoutC, output0, {node_name}__layoutC, &{node_name}__algo, {node_name}__workspace, {node_name}__wsSize, 0));')
        else:
            writer.wl(f'checkBlasErrors(cublasLtMatmul({node_name}__handle, {node_name}__desc, &{node_name}__alpha, input0 + {self.input_offset[0]}, {node_name}__layoutA, input1 + {self.input_offset[1]}, {node_name}__layoutB, &{node_name}__beta, input2, {node_name}__layoutBias, output0, {node_name}__layoutC, &{node_name}__algo, {node_name}__workspace, {node_name}__wsSize, 0));')
        writer.block_end()
        return writer.get_code() 

    def __str__(self) -> str:
        s = super().__str__()
        attrs = []
        attrs.append('input_perm=' + str(self.input_perm))
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
                 size_ba: int = -1, size_bb: int = -1, size_m: int = -1, size_n: int = -1, size_k: int = -1,
                 input0_perm: Tuple[int, int, int] = (0, 1, 2), input1_perm: Tuple[int, int, int] = (0, 1, 2), output_perm: Tuple[int, int, int] = (0, 1, 2),
                 alpha: float = 1.0, beta: float = 0.0, epilogue: Epilogue = Epilogue.CUBLASLT_EPILOGUE_DEFAULT,
                 input_offset: List[int] = [0, 0], output_offset: int = 0,) -> None:
        super().__init__(name, input_nodes, output_nodes, input_types, output_types, input_constants, size_ba, size_bb, size_m, size_n, size_k, input0_perm, input1_perm, output_perm, None, alpha, beta, epilogue, input_offset, output_offset)



class MatMul(UniMatMulNoBias):
    def __init__(
        self, name:str,
        input_nodes:List['IndexNode'], output_nodes:List[List['IndexNode']],
        input_types:List[TensorType], output_types:List[TensorType],
        input_constants: List[Optional[np.ndarray]],
    ) -> None:
        # super().__init__(name, input_nodes, output_nodes, input_types, output_types)
        if len(input_types[0].shape) > 3 or len(input_types[1].shape) > 3:
            raise NotImplementedError
        if len(input_types[0].shape) <= 2:
            size_ba = 1
        else:
            size_ba = input_types[0].shape[0]
        if len(input_types[1].shape) <= 2:
            size_bb = 1
        else:
            size_bb = input_types[1].shape[0]
        size_m = input_types[0].shape[-2]
        size_k = input_types[0].shape[-1]
        size_n = input_types[1].shape[-1]
        super().__init__(name, input_nodes, output_nodes, input_types, output_types, input_constants, size_ba, size_bb, size_m, size_n, size_k)


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
        attributes = {attr.name: attr for attr in onnx_node.attribute}
        alpha = attributes["alpha"].f
        beta = attributes["beta"].f
        transA = attributes["transA"].i if "transA" in attributes else 0
        transB = attributes["transB"].i if "transB" in attributes else 0
        if transA:
            size_k, size_m = input_types[0].shape
            input0_perm = [0, 2, 1]
        else:
            size_m, size_k = input_types[0].shape
            input0_perm = [0, 1, 2]
        if transB:
            size_n = input_types[1].shape[0]
            input1_perm = [0, 2, 1]
        else:
            size_n = input_types[1].shape[1]
            input1_perm = [0, 1, 2]
        bias_perm = [0, 1, 2]
        output_perm = [0, 1, 2]
        return cls(name, input_nodes, output_nodes, input_types, output_types, input_constants, 1, 1, size_m, size_n, size_k, input0_perm, input1_perm, output_perm, bias_perm, alpha=alpha, beta=beta)
