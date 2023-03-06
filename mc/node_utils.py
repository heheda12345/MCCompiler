import mc.operators as ops
import onnx
from typing import List, Optional
from mc.types import TensorType
from onnx import numpy_helper
import numpy as np

def parse_input_from_onnx(
    onnx_node: onnx.ValueInfoProto,
    ty: TensorType
):
    return ops.Input(name=onnx_node.name,
                    input_nodes=[],
                    output_nodes=[[]],
                    input_types=[],
                    output_types=[ty])

def parse_output_from_onnx(
    onnx_node: onnx.ValueInfoProto,
    ty: TensorType
):
    return ops.Output(name=onnx_node.name,
                    input_nodes=[None],
                    output_nodes=[],
                    input_types=[ty],
                    output_types=[])    

def parse_initializer_from_onnx(
    onnx_tensor: onnx.TensorProto,
    ty: TensorType
):
    return ops.Constant(name=onnx_tensor.name,
                    input_nodes=[],
                    output_nodes=[[]],
                    input_types=[],
                    output_types=[ty],
                    value=numpy_helper.to_array(onnx_tensor))

def parse_op_from_onnx(
    onnx_node: onnx.TensorProto,
    input_types: List[TensorType],
    output_types: List[TensorType],
    input_constants: List[Optional[np.ndarray]] = [],
):
    op_type = onnx_node.op_type
    if op_type not in dir(ops):
        raise NotImplementedError(f"Operator {op_type} is not supported yet.")
    op = getattr(ops, op_type)(
        name=onnx_node.name,
        input_nodes=[None] * len(input_types),
        output_nodes=[[] for _ in range(len(output_types))],
        input_types=input_types,
        output_types=output_types,
        input_constants=input_constants,
        onnx_node=onnx_node,
    )
    return op
