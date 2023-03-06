from mc.node import Node, IndexNode
from typing import List, Optional, Tuple
from mc.types import TensorType
import onnx
import numpy as np

class Softmax(Node):
    axis:int
    def __init__(
        self, name:str,
        input_nodes:List['IndexNode'], output_nodes:List[List['IndexNode']],
        input_types:List[TensorType], output_types:List[TensorType],
        input_constants: List[Optional[np.ndarray]],
        axis:int
    ) -> None:
        super().__init__(name, input_nodes, output_nodes, input_types, output_types, input_constants)
        self.axis = axis

    @classmethod
    def from_onnx(cls, name:str,
                    input_nodes: List['IndexNode'], output_nodes: List[List['IndexNode']],
                    input_types: List[TensorType], output_types: List[TensorType],
                    input_constants: List[np.ndarray], onnx_node: onnx.NodeProto) -> Node:
        return cls(name, input_nodes, output_nodes, input_types, output_types, input_constants,
                   axis=onnx_node.attribute[0].i)