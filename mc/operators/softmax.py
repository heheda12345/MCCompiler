from mc.node import Node, IndexNode
from typing import List, Optional, Tuple
from mc.types import TensorType
import onnx
import numpy as np
from mc.utils import CodeWriter, cpp_type

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


    def get_cuda_code(self, func_sig, node_name) -> str:
        if self.axis != -1 and self.axis != len(self.input_types[0].shape) - 1:
            raise NotImplementedError
        row_size = self.input_types[0].shape[-1]
        num_row = self.input_types[0].size() // row_size
        writer = CodeWriter()
        writer.wl(func_sig)
        writer.block_start()
        writer.wl(f"MCCompiler::SoftMax::softmax_last_col(input0, output0, {row_size}, {num_row});")
        writer.block_end()
        return writer.get_code()