from mc.node import Node, IndexNode
from typing import List, Optional, Tuple
from mc.types import TensorType
import onnx
import numpy as np

class LayerNorm(Node):
    eps:float
    gamma:np.ndarray
    beta:np.ndarray
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
