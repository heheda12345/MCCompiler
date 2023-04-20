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
    input0_stride: Tuple[int, int, int]
    input1_stride: Tuple[int, int, int]
    bias_stride: Tuple[int, int, int]
    output_stride: Tuple[int, int, int]
    alpha: float
    beta: float
    epilogue: Epilogue
    def __init__(
            self, name: str,
            input_nodes: List['IndexNode'], output_nodes: List[List['IndexNode']],
            input_types: List[TensorType], output_types: List[TensorType],
            input_constants: List[Optional[np.ndarray]],
            size_b: int = -1, size_m: int = -1, size_n: int = -1, size_k: int = -1,
            input0_stride: Tuple[int, int, int] = (0, 0, 0), input1_stride: Tuple[int, int, int] = (0, 0, 0),
            output_stride: Tuple[int, int, int] = (0, 0, 0),
            bias_stride: Optional[Tuple[int, int, int]]=None,
            alpha: float = 1.0, beta: float = 0.0,
            epilogue: Epilogue = Epilogue.CUBLASLT_EPILOGUE_DEFAULT,
        ) -> None:
        super().__init__(name, input_nodes, output_nodes, input_types, output_types, input_constants)
        self.size_b = size_b
        self.size_m = size_m
        self.size_n = size_n
        self.size_k = size_k
        self.input0_stride = input0_stride
        self.input1_stride = input1_stride
        self.output_stride = output_stride
        self.bias_stride = bias_stride
        self.alpha = alpha
        self.beta = beta
        self.epilogue = epilogue


    @classmethod
    def copy_attr_from(cls, node: 'UniMatMul', has_bias=False) -> 'UniMatMul':
        if has_bias:
            num_inputs = 3
        else:
            num_inputs = 2

        return cls(
            gen_name('UniMatMul'), [None] * num_inputs, [[]], [None] * num_inputs, [None], [],
            node.size_b, node.size_m, node.size_n, node.size_k,
            node.input0_stride, node.input1_stride, node.output_stride,
            node.alpha, node.beta, node.epilogue
        )

    def __str__(self) -> str:
        s = super().__str__()
        if self.epilogue != Epilogue.CUBLASLT_EPILOGUE_DEFAULT:
            s += f'(epilogue={self.epilogue.name})'
        return s


class MatMul(UniMatMul):
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
        input0_stride = [0, size_n, 1]
        if len(input_types[0].shape) == 3:
            input0_stride[0] = size_m * size_n
        input1_stride = [0, size_k, 1]
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
        return cls(name, input_nodes, output_nodes, input_types, output_types, input_constants, size_b, size_m, size_n, size_k, input0_stride, input1_stride, output_stride, bias_stride, alpha, beta)
