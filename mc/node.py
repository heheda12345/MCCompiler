from __future__ import annotations
from typing import List, Dict
from mc.types import TensorType
import onnx
import enum
import numpy as np

class NodeType(enum.Enum):
    input = 1
    initializer = 2
    node = 3
    output = 4


class Node:
    name: str
    input_nodes: List['IndexNode']
    output_nodes: List['IndexNode']

    input_types: List[TensorType]
    output_types: List[TensorType]

    def __init__(self, name:str='',
                 input_nodes:List['IndexNode']=[], output_nodes:List['IndexNode']=[],
                 input_types:List[TensorType]=[], output_types:List[TensorType]=[],
                 input_constants:List[np.ndarray]=[]) -> None:
        self.name = name
        self.input_nodes = input_nodes
        self.output_nodes = output_nodes
        self.input_types = input_types
        self.output_types = output_types

    def io_all_exist(self):
        for node in self.input_nodes:
            if node is None:
                return False
        for node in self.output_nodes:
            if node is None:
                return False
        return True

    def set_input(self, src_node: 'IndexNode', dst_node: 'IndexNode'):
        self.input_nodes[dst_node.index] = src_node
    
    def set_output(self, src_node: 'IndexNode', dst_node: 'IndexNode'):
        self.output_nodes[src_node.index] = dst_node
    
    def parse_op_from_onnx(self, onnx_node: onnx.NodeProto):
        raise NotImplementedError

    def __str__(self) -> str:
        return '{}: {} -> {} | {} -> {}'.format(
            self.name, self.input_nodes, self.output_nodes, self.input_types, self.output_types
        )

    def __repr__(self) -> str:
        return self.__str__()


class IndexNode:
    index: int
    node: Node

    def __init__(self, node, index) -> None:
        self.node = node
        self.index = index
    
    def __str__(self) -> str:
        # print("str idxnode called")
        return '{}[{}]'.format(self.node.name, self.index)

    def __repr__(self) -> str:
        return self.__str__()