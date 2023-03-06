from mc.node import Node, IndexNode
from typing import List, Optional, Tuple
from mc.types import TensorType
import onnx
import numpy as np

class Transpose(Node):
    perm: List[int]
    def __init__(
        self, name:str,
        input_nodes:List['IndexNode'], output_nodes:List[List['IndexNode']],
        input_types:List[TensorType], output_types:List[TensorType],
        input_constants: List[Optional[np.ndarray]] = [],
        perm:Optional[List[int]] = None,
    ) -> None:
        super().__init__(name, input_nodes, output_nodes, input_types, output_types, input_constants)
        self.perm = perm
        
    @classmethod
    def from_onnx(cls, name:str,
                    input_nodes: List['IndexNode'], output_nodes: List[List['IndexNode']],
                    input_types: List[TensorType], output_types: List[TensorType],
                    input_constants: List[np.ndarray],
                    onnx_node:Optional[onnx.NodeProto] = None,
    ) -> None:
        perm = list(onnx_node.attribute[0].ints)
        return cls(name, input_nodes, output_nodes, input_types, output_types, input_constants, perm)
