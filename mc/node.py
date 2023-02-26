from __future__ import annotations
from typing import List, Optional
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
    type: NodeType

    @classmethod
    def from_onnx(cls, onnx_node: onnx.NodeProto | onnx.ValueInfoProto | onnx.TensorProto, node_type):
        op = cls()
        op.name = onnx_node.name
        op.type = node_type
        if node_type is NodeType.input or node_type is NodeType.initializer:
            op.input_nodes = []
            op.output_nodes = [None]
        elif node_type is NodeType.node:
            op.input_nodes = [None] * len(onnx_node.input)
            op.output_nodes = [None] * len(onnx_node.output)
        elif node_type is NodeType.output:
            op.input_nodes = [None]
            op.output_nodes = []
        return op
    
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

    def __str__(self) -> str:
        return '{}: {} -> {}'.format(self.name, self.input_nodes, self.output_nodes)

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