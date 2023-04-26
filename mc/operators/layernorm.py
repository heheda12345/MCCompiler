from mc.node import Node, IndexNode
from typing import List, Optional, Tuple, Dict
from mc.types import TensorType
import onnx
import numpy as np
from mc.utils import CodeWriter, cpp_type

class LayerNorm(Node):
    eps:float
    gamma:np.ndarray
    beta:np.ndarray
    gamma_name: Optional[str]
    beta_name: Optional[str]
    def __init__(
        self, name:str,
        input_nodes:List['IndexNode'], output_nodes:List[List['IndexNode']],
        input_types:List[TensorType], output_types:List[TensorType],
        input_constants:List[Optional[np.ndarray]],
        eps:float,
        gamma:np.ndarray,
        beta:np.ndarray,
    ) -> None:
        super().__init__(name, input_nodes, output_nodes, input_types, output_types, input_constants)
        self.eps = eps
        self.gamma = gamma
        self.beta = beta
        self.gamma_name = None
        self.beta_name = None

    def get_constant_tensors(self, node_name) -> Dict[str, np.ndarray]:
        self.gamma_name = f"{node_name}__gamma"
        self.beta_name = f"{node_name}__beta"
        return {
            self.gamma_name: self.gamma,
            self.beta_name: self.beta,
        }

    def get_cuda_code(self, func_sig, node_name) -> str:
        if self.gamma_name is None or self.beta_name is None:
            raise ValueError("gamma_name or beta_name is None")
        writer = CodeWriter()
        writer.wl(func_sig)
        writer.block_start()
        row_size = self.input_types[0].shape[-1]
        cpp_dtype = cpp_type(self.input_types[0].dtype)
        writer.write(f"MCCompiler::LayerNorm::layernorm<{row_size}, {cpp_dtype}>(input0, {self.gamma_name}, {self.beta_name}, output0, (int) {self.input_types[0].size() // row_size}, ({cpp_dtype}) {self.eps});")
        writer.block_end()
        return writer.get_code()
