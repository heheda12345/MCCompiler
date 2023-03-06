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
    output_nodes: List[List['IndexNode']]

    input_types: List[TensorType]
    output_types: List[TensorType]

    input_constants: List[np.ndarray]

    def __init__(self, name:str='',
                 input_nodes:List['IndexNode']=[], output_nodes:List[List['IndexNode']]=[],
                 input_types:List[TensorType]=[], output_types:List[TensorType]=[],
                 input_constants:List[np.ndarray]=[]) -> None:
        self.name = name
        self.input_nodes = input_nodes
        self.output_nodes = output_nodes
        self.input_types = input_types
        self.output_types = output_types
        self.input_constants = input_constants

    def input_exist(self):
        for node in self.input_nodes:
            if node is None:
                return False
        return True

    def set_input(self, src_node: 'IndexNode', dst_node: 'IndexNode', allow_replace=False):
        if not allow_replace:
            assert self.input_nodes[dst_node.index] is None
        self.input_nodes[dst_node.index] = src_node
    
    def add_output(self, src_node: 'IndexNode', dst_node: 'IndexNode'):
        self.output_nodes[src_node.index].append(dst_node)
    
    def parse_op_from_onnx(self, onnx_node: onnx.NodeProto):
        raise NotImplementedError

    def __str__(self) -> str:
        return '{}({}): {} -> {} | {} -> {}'.format(
            self.name, self.__class__.__name__, self.input_nodes, self.output_nodes, self.input_types, self.output_types
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