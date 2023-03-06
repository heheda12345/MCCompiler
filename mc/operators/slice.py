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

    def __init__(self, name:str,
        input_nodes:List['IndexNode'], output_nodes:List[List['IndexNode']],
        input_types:List[TensorType], output_types:List[TensorType],
        input_constants: List[Optional[np.ndarray]],
        starts: List[int],
        ends: List[int],
        axes: List[int],
        steps: List[int],
        ) -> None:
        super().__init__(name, input_nodes, output_nodes, input_types, output_types, input_constants)
        self.starts = starts
        self.ends = ends
        self.axes = axes
        self.steps = steps


    @classmethod
    def from_onnx(
        cls, name:str,
        input_nodes:List['IndexNode'], output_nodes:List[List['IndexNode']],
        input_types:List[TensorType], output_types:List[TensorType],
        input_constants: List[Optional[np.ndarray]],
        onnx_node:onnx.NodeProto,
    ) -> None:
        if len(input_constants) == 0:
            raise NotImplementedError
        else:
            starts = input_constants[1].tolist()
            ends = input_constants[2].tolist()
            axes = input_constants[3].tolist() if len(input_constants) > 3 else [i for i in range(len(starts))]
            steps = input_constants[4].tolist() if len(input_constants) > 4 else [1 for i in range(len(starts))]
        axes = [i if i >= 0 else i + len(input_types[0].shape) for i in axes]
        return cls(name, input_nodes, output_nodes, input_types, output_types, input_constants, starts, ends, axes, steps)

