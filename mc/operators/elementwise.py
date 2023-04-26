from mc.node import Node, IndexNode
from typing import List, Optional, Union
from mc.types import TensorType
import onnx
import numpy as np
from mc.utils import CodeWriter

class ElementWise(Node):
    pass

class ElementWiseUnary(ElementWise):
    pass

class ElementWiseBinary(ElementWise):
    pass

class AddUni(ElementWiseUnary):
    value: np.ndarray
    def __init__(
        self, name:str,
        input_nodes:List['IndexNode'], output_nodes:List[List['IndexNode']],
        input_types:List[TensorType], output_types:List[TensorType],
        input_constants: List[Optional[np.ndarray]],
        value: np.ndarray
    ) -> None:
        super().__init__(name, input_nodes, output_nodes, input_types, output_types, input_constants)
        self.value = value
    
    @classmethod
    def unused_onnx_inputs(cls):
        return [1]


class Add(ElementWiseBinary):
    @classmethod
    def from_onnx(cls, name:str,
                    input_nodes: List['IndexNode'], output_nodes: List[List['IndexNode']],
                    input_types: List[TensorType], output_types: List[TensorType],
                    input_constants: List[np.ndarray], onnx_node: onnx.NodeProto) -> Node:
        if input_constants[1] is not None and input_constants[1].size == 1:
            return AddUni(name, input_nodes, output_nodes, input_types, output_types, input_constants, input_constants[1])
        return cls(name, input_nodes, output_nodes, input_types, output_types, input_constants)

    def get_cuda_code(self, func_sig, node_name) -> str:
        writer = CodeWriter()
        writer.wl(func_sig)
        writer.block_start()
        writer.wl(f"MCCompiler::element_wise::add(input0, input1, output0, {self.output_types[0].size()});")
        writer.block_end()
        return writer.get_code()


class Sub(ElementWiseBinary):
    pass

class MulUni(ElementWiseUnary):
    value: np.ndarray
    def __init__(
        self, name:str,
        input_nodes:List['IndexNode'], output_nodes:List[List['IndexNode']],
        input_types:List[TensorType], output_types:List[TensorType],
        input_constants: List[Optional[np.ndarray]],
        value: np.ndarray
    ) -> None:
        super().__init__(name, input_nodes, output_nodes, input_types, output_types, input_constants)
        self.value = value
    
    @classmethod
    def unused_onnx_inputs(cls):
        return [1]


class Mul(ElementWiseBinary):
    @classmethod
    def from_onnx(cls, name:str,
                    input_nodes: List['IndexNode'], output_nodes: List[List['IndexNode']],
                    input_types: List[TensorType], output_types: List[TensorType],
                    input_constants: List[np.ndarray], onnx_node: onnx.NodeProto) -> Node:
        if input_constants[1] is not None and input_constants[1].size == 1:
            return MulUni(name, input_nodes, output_nodes, input_types, output_types, input_constants, input_constants[1])
        return cls(name, input_nodes, output_nodes, input_types, output_types, input_constants)


class DivUni(ElementWiseUnary):
    value: np.ndarray
    def __init__(
        self, name:str,
        input_nodes:List['IndexNode'], output_nodes:List[List['IndexNode']],
        input_types:List[TensorType], output_types:List[TensorType],
        input_constants: List[Optional[np.ndarray]],
        value: np.ndarray
    ) -> None:
        super().__init__(name, input_nodes, output_nodes, input_types, output_types, input_constants)
        self.value = value
    
    @classmethod
    def unused_onnx_inputs(cls):
        return [1]


class Div(ElementWiseBinary):
    @classmethod
    def from_onnx(cls, name:str,
                    input_nodes: List['IndexNode'], output_nodes: List[List['IndexNode']],
                    input_types: List[TensorType], output_types: List[TensorType],
                    input_constants: List[np.ndarray], onnx_node: onnx.NodeProto) -> Node:
        if input_constants[1] is not None and input_constants[1].size == 1:
            return DivUni(name, input_nodes, output_nodes, input_types, output_types, input_constants, input_constants[1])
        return cls(name, input_nodes, output_nodes, input_types, output_types, input_constants)


class PowUni(ElementWiseUnary):
    value: np.ndarray
    def __init__(
        self, name:str,
        input_nodes:List['IndexNode'], output_nodes:List[List['IndexNode']],
        input_types:List[TensorType], output_types:List[TensorType],
        input_constants: List[Optional[np.ndarray]],
        value: np.ndarray
    ) -> None:
        super().__init__(name, input_nodes, output_nodes, input_types, output_types, input_constants)
        self.value = value
    
    @classmethod
    def unused_onnx_inputs(cls):
        return [1]

class Pow(ElementWiseBinary):
    @classmethod
    def from_onnx(cls, name:str,
                    input_nodes: List['IndexNode'], output_nodes: List[List['IndexNode']],
                    input_types: List[TensorType], output_types: List[TensorType],
                    input_constants: List[np.ndarray], onnx_node: onnx.NodeProto) -> Node:
        if input_constants[1] is not None and input_constants[1].size == 1:
            return PowUni(name, input_nodes, output_nodes, input_types, output_types, input_constants, input_constants[1])
        return cls(name, input_nodes, output_nodes, input_types, output_types, input_constants)

class Equal(ElementWiseBinary):
    pass

class Sqrt(ElementWiseUnary):
    pass

class Erf(ElementWiseUnary):
    pass

class GELU(ElementWiseUnary):
    pass

class Where(ElementWise):
    def get_cuda_code(self, func_sig, node_name) -> str:
        writer = CodeWriter()
        writer.wl(func_sig)
        writer.block_start()
        writer.wl(f"MCCompiler::element_wise::where(input0, input1, input2, output0, {self.output_types[0].size()});")
        writer.block_end()
        return writer.get_code()

class Cast(ElementWise):
    pass