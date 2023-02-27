from mc.node import Node, IndexNode
from typing import List, Optional, Tuple
from mc.types import TensorType
import onnx

class ReduceSum(Node):
    axes:List[int]
    keepdims:bool
    def __init__(
        self, name:str,
        input_nodes:List['IndexNode'], output_nodes:List['IndexNode'],
        input_types:List[TensorType], output_types:List[TensorType],
        onnx_node:Optional[onnx.NodeProto] = None,
    ) -> None:
        super().__init__(name, input_nodes, output_nodes, input_types, output_types)
        attributes = {attr.name: attr for attr in onnx_node.attribute}
        self.axes = list(attributes['axes'].ints)
        self.keepdims = attributes['keepdims'].i