from __future__ import annotations
from enum import Enum
from typing import List
import onnx


# https://github.com/onnx/onnx/blob/main/docs/IR.md#tensor-element-types
# Type = Enum('unknown', 'float16', 'float32', 'float64', 'int8', 'int16', 'int32', 'int64', 'uint8', 'uint16', 'uint32', 'uint64', 'bool', 'complex64', 'complex128')
class Type(Enum):
    UNKNOWN = 0
    FLOAT16 = 1
    FLOAT32 = 2
    FLOAT64 = 3
    INT8 = 4
    INT16 = 5
    INT32 = 6
    INT64 = 7
    UINT8 = 8
    UINT16 = 9
    UINT32 = 10
    UINT64 = 11
    BOOL = 12
    COMPLEX64 = 13
    COMPLEX128 = 14

    @classmethod
    def from_onnx(cls, onnx_dtype: onnx.TensorProto.DataType):
        if onnx_dtype == onnx.TensorProto.FLOAT16: return cls.FLOAT16
        if onnx_dtype == onnx.TensorProto.FLOAT: return cls.FLOAT32
        if onnx_dtype == onnx.TensorProto.DOUBLE: return cls.FLOAT64
        if onnx_dtype == onnx.TensorProto.INT8: return cls.INT8
        if onnx_dtype == onnx.TensorProto.INT16: return cls.INT16
        if onnx_dtype == onnx.TensorProto.INT32: return cls.INT32
        if onnx_dtype == onnx.TensorProto.INT64: return cls.INT64
        if onnx_dtype == onnx.TensorProto.UINT8: return cls.UINT8
        if onnx_dtype == onnx.TensorProto.UINT16: return cls.UINT16
        if onnx_dtype == onnx.TensorProto.UINT32: return cls.UINT32
        if onnx_dtype == onnx.TensorProto.UINT64: return cls.UINT64
        if onnx_dtype == onnx.TensorProto.BOOL: return cls.BOOL
        if onnx_dtype == onnx.TensorProto.COMPLEX64: return cls.COMPLEX64
        if onnx_dtype == onnx.TensorProto.COMPLEX128: return cls.COMPLEX128
        raise ValueError('Unknown ONNX dtype: {}'.format(onnx_dtype))


class TensorType:
    shape: List[int]
    dtype: Type

    def __init__(self, shape: List[int], dtype: Type) -> None:
        self.shape = shape
        self.dtype = dtype

    @classmethod
    def from_onnx_type(cls, onnx_type: onnx.TypeProto) -> TensorType:
        assert onnx_type.HasField('tensor_type')
        shape = []
        for dim in onnx_type.tensor_type.shape.dim:
            shape.append(dim.dim_value)
        onnx_ty = Type.from_onnx(onnx_type.tensor_type.elem_type)
        out = cls(shape, onnx_ty)
        return out

    @classmethod
    def from_onnx_tensor(cls, onnx_value_info: onnx.TensorProto) -> TensorType:
        shape = []
        for dim in onnx_value_info.dims:
            shape.append(dim)
        onnx_ty = Type.from_onnx(onnx_value_info.data_type)
        out = cls(shape, onnx_ty)
        return out

    def __str__(self) -> str:
        return '({}, {})'.format(self.shape, self.dtype)
    
    def __repr__(self) -> str:
        return self.__str__()
