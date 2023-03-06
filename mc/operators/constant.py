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
        input_nodes:List['IndexNode'], output_nodes:List['IndexNode'],
        input_types:List[TensorType], output_types:List[TensorType],
        input_constants: List[Optional[np.ndarray]] = [],
        onnx_node:Optional[onnx.NodeProto] = None,
        value:Optional[np.ndarray] = None,
    ) -> None:
        super().__init__(name, input_nodes, output_nodes, input_types, output_types, input_constants)
        value_is_in_constants = len(input_constants) > 0 and input_constants[0] is not None
        assert (onnx_node is not None) + (value is not None) + value_is_in_constants == 1
        if onnx_node is not None:
            self.value = numpy_helper.to_array(onnx_node.attribute[0].t)
        elif value_is_in_constants:
            self.value = input_constants[0]
        else:
            self.value = value

class ConstantOfShape(Constant):
    def __init__(
        self, name:str,
        input_nodes:List['IndexNode'], output_nodes:List['IndexNode'],
        input_types:List[TensorType], output_types:List[TensorType],
        input_constants: List[Optional[np.ndarray]] = [],
        onnx_node:Optional[onnx.NodeProto] = None,
    ) -> None:
        super().__init__(name, input_nodes, output_nodes, input_types, output_types, input_constants)