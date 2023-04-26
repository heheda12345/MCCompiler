from mc.node import Node, IndexNode
from typing import List, Optional
from mc.types import TensorType
import onnx
import numpy as np
from mc.utils import CodeWriter

class Expand(Node):
    @classmethod
    def unused_onnx_inputs(cls):
        return [1]

    def get_cuda_code(self, func_sig, node_name) -> str:
        writer = CodeWriter()
        writer.wl(func_sig)
        writer.block_start()
        writer.wl(f"MCCompiler::element_wise::expand(input0, output0, {self.input_types[0].size()}, {self.output_types[0].size() // self.input_types[0].size()});")
        writer.block_end()
        return writer.get_code()
