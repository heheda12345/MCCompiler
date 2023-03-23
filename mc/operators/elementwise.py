from mc.node import Node, IndexNode
from typing import List, Optional, Union
from mc.types import TensorType
import onnx
import numpy as np

class ElementWise(Node):
    pass

class ElementWiseUnary(ElementWise):
    pass

class ElementWiseBinary(ElementWise):
    pass

class Add(ElementWiseBinary):
    pass

class Sub(ElementWiseBinary):
    pass

class Mul(ElementWiseBinary):
    pass

class Div(ElementWiseBinary):
    pass

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
        if input_constants[1] is not None:
            return PowUni(name, input_nodes, output_nodes, input_types, output_types, input_constants, input_constants[1])
        return cls(name, input_nodes, output_nodes, input_types, output_types, input_constants)

class Equal(ElementWiseBinary):
    pass

class Sqrt(ElementWiseUnary):
    pass

class Erf(ElementWiseUnary):
    pass

class Where(ElementWise):
    pass

class Cast(ElementWise):
    pass