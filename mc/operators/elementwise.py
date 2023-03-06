from mc.node import Node, IndexNode
from typing import List, Optional
from mc.types import TensorType
import onnx
import numpy as np

class ElementWise(Node):
    pass

class ElementWiseUnary(ElementWise):
    pass

class ElementWiseBinary(ElementWise):
    pass

class Add(ElementWiseBinary):
    pass

class Sub(ElementWiseBinary):
    pass

class Mul(ElementWiseBinary):
    pass

class Div(ElementWiseBinary):
    pass

class Pow(ElementWiseBinary):
    pass

class Equal(ElementWiseBinary):
    pass

class Sqrt(ElementWiseUnary):
    pass

class Erf(ElementWiseUnary):
    pass

class Where(ElementWise):
    pass

class Cast(ElementWise):
    pass