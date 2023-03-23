from mc.node import Node, IndexNode
from typing import List, Optional, Tuple
from mc.types import TensorType
import onnx
import numpy as np

class Reshape(Node):
    @classmethod
    def unused_onnx_inputs(cls):
        return [1]