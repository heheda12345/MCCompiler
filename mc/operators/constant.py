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
        onnx_node:Optional[onnx.NodeProto] = None,
        value:Optional[np.ndarray] = None,
    ) -> None:
        super().__init__(name, input_nodes, output_nodes, input_types, output_types)
        assert (onnx_node is not None) + (value is not None) == 1
        if onnx_node is not None:
            self.value = numpy_helper.to_array(onnx_node.attribute[0].t)
        else:
            self.value = value

class ConstantOfShape(Constant):
    def __init__(
        self, name:str,
        input_nodes:List['IndexNode'], output_nodes:List['IndexNode'],
        input_types:List[TensorType], output_types:List[TensorType],
        onnx_node:Optional[onnx.NodeProto] = None,
    ) -> None:
        value = numpy_helper.to_array(onnx_node.attribute[0].t)
        value = np.full(output_types[0].shape, value)
        super().__init__(name, input_nodes, output_nodes, input_types, output_types, value=value)