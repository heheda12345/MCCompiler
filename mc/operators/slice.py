from mc.node import Node, IndexNode
from typing import List, Optional
from mc.types import TensorType
import onnx
import numpy as np

class Slice(Node):
    starts: List[int]
    ends: List[int]
    axes: List[int]
    steps: List[int]
    def __init__(
        self, name:str,
        input_nodes:List['IndexNode'], output_nodes:List[List['IndexNode']],
        input_types:List[TensorType], output_types:List[TensorType],
        input_constants: List[Optional[np.ndarray]] = [],
        onnx_node:Optional[onnx.NodeProto] = None,
    ) -> None:
        super().__init__(name, input_nodes, output_nodes, input_types, output_types, input_constants)
        if len(input_constants) == 0:
            raise NotImplementedError
        else:
            self.starts = input_constants[1].tolist()
            self.ends = input_constants[2].tolist()
            self.axes = input_constants[3].tolist() if len(input_constants) > 3 else [i for i in range(len(self.starts))]
            self.steps = input_constants[4].tolist() if len(input_constants) > 4 else [1 for i in range(len(self.starts))]
        self.axes = [i if i >= 0 else i + len(input_types[0].shape) for i in self.axes]
