from __future__ import annotations
from enum import Enum
from typing import List
import onnx
import numpy as np


class TensorType:
    shape: List[int]
    dtype: np.dtype

    def __init__(self, shape: List[int], dtype: np.dtype) -> None:
        self.shape = shape
        self.dtype = dtype

    @classmethod
    def from_onnx_type(cls, onnx_type: onnx.TypeProto) -> TensorType:
        assert onnx_type.HasField('tensor_type')
        shape = []
        for dim in onnx_type.tensor_type.shape.dim:
            shape.append(dim.dim_value)
        onnx_ty = onnx.helper.tensor_dtype_to_np_dtype(onnx_type.tensor_type.elem_type)
        out = cls(shape, onnx_ty)
        return out

    @classmethod
    def from_onnx_tensor(cls, onnx_value_info: onnx.TensorProto) -> TensorType:
        shape = []
        for dim in onnx_value_info.dims:
            shape.append(dim)
        onnx_ty = onnx.helper.tensor_dtype_to_np_dtype(onnx_value_info.data_type)
        out = cls(shape, onnx_ty)
        return out

    def size(self) -> int:
        return int(np.prod(self.shape))

    def __str__(self) -> str:
        return '({}, {})'.format(self.shape, self.dtype)
    
    def __repr__(self) -> str:
        return self.__str__()
