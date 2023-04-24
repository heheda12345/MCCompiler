from mc.node import Node, IndexNode, gen_name
from typing import List, Optional, Tuple
from mc.types import TensorType
import onnx
import numpy as np
import enum

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
        print("matmul: alpha=", alpha, name)
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
        self.input_offset = input_offset
        self.output_offset = output_offset


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
        if len(input_types[0].shape) > 3 or len(input_types[1].shape) > 3:
            raise NotImplementedError
        size_b = max(input_types[0].shape[0], input_types[1].shape[0])
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
