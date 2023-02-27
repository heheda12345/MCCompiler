from __future__ import annotations
from typing import List
from mc.types import TensorType
import onnx
import enum

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
    type: NodeType

    @classmethod
    def from_onnx(cls, onnx_node: onnx.NodeProto | onnx.ValueInfoProto | onnx.TensorProto,
                  node_type: NodeType):
        node = cls()
        node.name = onnx_node.name
        node.type = node_type
        if node_type is NodeType.input or node_type is NodeType.initializer:
            node.input_nodes = []
            node.output_nodes = [None]
        elif node_type is NodeType.node:
            node.input_nodes = [None] * len(onnx_node.input)
            node.output_nodes = [None] * len(onnx_node.output)
        elif node_type is NodeType.output:
            node.input_nodes = [None]
            node.output_nodes = []
        return node
    
    def io_all_exist(self):
        for node in self.input_nodes:
            if node is None:
                return False
        for node in self.output_nodes:
            if node is None:
                return False
        return True

    def set_io_type(self, input_types: List[TensorType], output_types: List[TensorType]):
        self.input_types = input_types
        self.output_types = output_types

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