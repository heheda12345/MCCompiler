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
        input_constants: List[Optional[np.ndarray]] = [],
        onnx_node:Optional[onnx.NodeProto] = None,
        keepdims:Optional[bool] = None,
        axes:Optional[List[int]] = None,
        type:ReduceType = ReduceType.UNKNOWN,
    ) -> None:
        super().__init__(name, input_nodes, output_nodes, input_types, output_types, input_constants)
        assert (onnx_node is not None) + (keepdims is not None and axes is not None) == 1
        if onnx_node is not None:
            attributes = {attr.name: attr for attr in onnx_node.attribute}
            self.axes = list(attributes['axes'].ints)
            if 'keepdims' in attributes:
                self.keepdims = attributes['keepdims'].i
            else:
                self.keepdims = True
        else:
            self.keepdims = keepdims
            self.axes = axes
        self.type = type


class ReduceSum(Reduce):
    def __init__(
        self, name:str,
        input_nodes:List['IndexNode'], output_nodes:List[List['IndexNode']],
        input_types:List[TensorType], output_types:List[TensorType],
        input_constants: List[Optional[np.ndarray]] = [],
        onnx_node:Optional[onnx.NodeProto] = None,
        keepdims:Optional[bool] = None,
        axes:Optional[List[int]] = None,
    ) -> None:
        super().__init__(name, input_nodes, output_nodes, input_types, output_types, input_constants, onnx_node, keepdims, axes, ReduceType.SUM)


class ReduceMean(Reduce):
    axes:List[int]
    keepdims:bool
    def __init__(
        self, name:str,
        input_nodes:List['IndexNode'], output_nodes:List[List['IndexNode']],
        input_types:List[TensorType], output_types:List[TensorType],
        input_constants:List[Optional[np.ndarray]] = [],
        onnx_node:Optional[onnx.NodeProto] = None,
        keepdims:Optional[bool] = None,
        axes:Optional[List[int]] = None,
    ) -> None:
        super().__init__(name, input_nodes, output_nodes, input_types, output_types, input_constants, onnx_node, keepdims, axes, ReduceType.MEAN)