from mc.node import Node, IndexNode
from typing import List, Optional, Tuple
from mc.types import TensorType
import onnx
import enum
import numpy as np

class ReduceType(enum.Enum):
    UNKNOWN = -1
    SUM = 0
    MEAN = 1


class Reduce(Node):
    axes:List[int]
    keepdims:bool
    type: ReduceType
    
    def __init__(
        self, name:str,
        input_nodes:List['IndexNode'], output_nodes:List['IndexNode'],
        input_types:List[TensorType], output_types:List[TensorType],
        input_constants: List[Optional[np.ndarray]] = [],
        onnx_node:Optional[onnx.NodeProto] = None,
        type:ReduceType = ReduceType.UNKNOWN,
    ) -> None:
        super().__init__(name, input_nodes, output_nodes, input_types, output_types, input_constants)
        attributes = {attr.name: attr for attr in onnx_node.attribute}
        self.axes = list(attributes['axes'].ints)
        if 'keepdims' in attributes:
            self.keepdims = attributes['keepdims'].i
        else:
            self.keepdims = True
        self.type = type


class ReduceSum(Reduce):
    def __init__(
        self, name:str,
        input_nodes:List['IndexNode'], output_nodes:List['IndexNode'],
        input_types:List[TensorType], output_types:List[TensorType],
        input_constants: List[Optional[np.ndarray]] = [],
        onnx_node:Optional[onnx.NodeProto] = None,
    ) -> None:
        super().__init__(name, input_nodes, output_nodes, input_types, output_types, input_constants, onnx_node, ReduceType.SUM)


class ReduceMean(Reduce):
    axes:List[int]
    keepdims:bool
    def __init__(
        self, name:str,
        input_nodes:List['IndexNode'], output_nodes:List['IndexNode'],
        input_types:List[TensorType], output_types:List[TensorType],
        input_constants: List[Optional[np.ndarray]] = [],
        onnx_node:Optional[onnx.NodeProto] = None,
    ) -> None:
        super().__init__(name, input_nodes, output_nodes, input_types, output_types, input_constants, onnx_node, ReduceType.MEAN)