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
    axes: List[int]
    keepdims: bool
    type: ReduceType
    
    def __init__(
        self, name:str,
        input_nodes:List['IndexNode'], output_nodes:List[List['IndexNode']],
        input_types:List[TensorType], output_types:List[TensorType],
        input_constants: List[Optional[np.ndarray]],
        keepdims:bool,
        axes:List[int],
        type:ReduceType = ReduceType.UNKNOWN,
    ) -> None:
        super().__init__(name, input_nodes, output_nodes, input_types, output_types, input_constants)
        self.keepdims = keepdims
        self.axes = axes
        self.type = type
    
    @classmethod
    def from_onnx(cls, name:str,
        input_nodes:List['IndexNode'], output_nodes:List[List['IndexNode']],
        input_types:List[TensorType], output_types:List[TensorType],
        input_constants: List[Optional[np.ndarray]],
        onnx_node:onnx.NodeProto):
        attributes = {attr.name: attr for attr in onnx_node.attribute}
        axes = list(attributes['axes'].ints)        
        keepdims = attributes['keepdims'].i if 'keepdims' in attributes else True
        return cls(name, input_nodes, output_nodes, input_types, output_types, input_constants, keepdims, axes)


class ReduceSum(Reduce):
    def __init__(
        self, name:str,
        input_nodes:List['IndexNode'], output_nodes:List[List['IndexNode']],
        input_types:List[TensorType], output_types:List[TensorType],
        input_constants: List[Optional[np.ndarray]],
        keepdims:bool,
        axes:List[int],
    ) -> None:
        super().__init__(name, input_nodes, output_nodes, input_types, output_types, input_constants, keepdims, axes, ReduceType.SUM)


class ReduceMean(Reduce):
    axes:List[int]
    keepdims:bool
    def __init__(
        self, name:str,
        input_nodes:List['IndexNode'], output_nodes:List[List['IndexNode']],
        input_types:List[TensorType], output_types:List[TensorType],
        input_constants:List[Optional[np.ndarray]],
        keepdims:bool,
        axes:List[int],
    ) -> None:
        super().__init__(name, input_nodes, output_nodes, input_types, output_types, input_constants, keepdims, axes, ReduceType.MEAN)