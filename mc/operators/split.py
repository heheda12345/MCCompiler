from mc.node import Node, IndexNode
from typing import List, Optional
from mc.types import TensorType
import onnx
import numpy as np

class Split(Node):
    axis: int
    split: List[int]
    def __init__(
        self, name:str,
        input_nodes:List['IndexNode'], output_nodes:List[List['IndexNode']],
        input_types:List[TensorType], output_types:List[TensorType],
        input_constants: List[Optional[np.ndarray]] = [],
        onnx_node:Optional[onnx.NodeProto] = None,
    ) -> None:
        super().__init__(name, input_nodes, output_nodes, input_types, output_types, input_constants)
        self.axis = onnx_node.attribute[0].i
        if self.axis < 0:
            self.axis += len(input_types[0].shape)
        self.split = list(onnx_node.attribute[1].ints)
