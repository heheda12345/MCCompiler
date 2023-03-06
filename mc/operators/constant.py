from mc.node import Node, IndexNode
from typing import List, Optional
from mc.types import TensorType
import onnx
import numpy as np
from onnx import numpy_helper

class Constant(Node):
    value: np.ndarray
    def __init__(
        self, name:str,
        input_nodes:List['IndexNode'], output_nodes:List[List['IndexNode']],
        input_types:List[TensorType], output_types:List[TensorType],
        input_constants: List[Optional[np.ndarray]],
        value:Optional[np.ndarray],
    ) -> None:
        super().__init__(name, input_nodes, output_nodes, input_types, output_types, input_constants)
        value_is_in_constants = len(input_constants) > 0 and input_constants[0] is not None
        assert value is not None
        self.value = value
    
    @classmethod
    def from_onnx(cls, name:str,
                  input_nodes: List['IndexNode'], output_nodes: List[List['IndexNode']],
                  input_types: List[TensorType], output_types: List[TensorType],
                  input_constants: List[np.ndarray], onnx_node: onnx.NodeProto) -> Node:
        value = numpy_helper.to_array(onnx_node.attribute[0].t)
        return cls(name, input_nodes, output_nodes, input_types, output_types, input_constants, value=value)


class ConstantOfShape(Constant):
    def __init__(
        self, name:str,
        input_nodes:List['IndexNode'], output_nodes:List[List['IndexNode']],
        input_types:List[TensorType], output_types:List[TensorType],
        input_constants: List[Optional[np.ndarray]]
    ) -> None:
        super().__init__(name, input_nodes, output_nodes, input_types, output_types, input_constants)
    
    @classmethod
    def from_onnx(cls, name:str,
                  input_nodes: List['IndexNode'], output_nodes: List[List['IndexNode']],
                  input_types: List[TensorType], output_types: List[TensorType],
                  input_constants: List[np.ndarray], onnx_node: onnx.NodeProto) -> Node:
        raise NotImplementedError