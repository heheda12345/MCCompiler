from __future__ import annotations
from typing import List, Dict, Set, Tuple
from mc.types import TensorType
import onnx
import enum
import numpy as np

class NodeType(enum.Enum):
    input = 1
    initializer = 2
    node = 3
    output = 4


node_name_used: Set[str] = set()

def gen_name(prefix: str) -> str:
    i = 0
    while True:
        name = f'{prefix}.{i}'
        if name not in node_name_used:
            return name
        i += 1

class Node:
    name: str
    input_nodes: List['IndexNode']
    output_nodes: List[List['IndexNode']]

    input_types: List[TensorType]
    output_types: List[TensorType]

    input_constants: List[np.ndarray]

    def __init__(self, name:str,
                 input_nodes:List['IndexNode'], output_nodes:List[List['IndexNode']],
                 input_types:List[TensorType], output_types:List[TensorType],
                 input_constants:List[np.ndarray]) -> None:
        if name in node_name_used:
            raise ValueError(f'Node name {name} is already used')
        node_name_used.add(name)
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
    
    def remove_output_edge(self, output_idx: int, dst_name: str):
        prev_size = len(self.output_nodes[output_idx])
        self.output_nodes[output_idx] = [idx_node for idx_node in self.output_nodes[output_idx] if idx_node.node.name != dst_name]
        if prev_size == len(self.output_nodes[output_idx]):
            raise ValueError(f'{dst_name} not found in the {output_idx}-th output of {self}')
    
    def remove_input(self, input_idx: int):
        assert input_idx == len(self.input_nodes) - 1, 'Only support remove last input now'
        self.input_nodes.pop()

    def parse_op_from_onnx(self, onnx_node: onnx.NodeProto):
        raise NotImplementedError

    def __str__(self) -> str:
        return '{}({}): {} -> {}'.format(
            self.name, self.__class__.__name__, self.input_nodes, self.output_nodes
        )
        # return '{}({}): {} -> {} | {} -> {}'.format(
            # self.name, self.__class__.__name__, self.input_nodes, self.output_nodes, self.input_types, self.output_types
        # )

    def __repr__(self) -> str:
        return self.__str__()
    
    @classmethod
    def from_onnx(cls, name:str,
                 input_nodes:List['IndexNode'], output_nodes:List[List['IndexNode']],
                 input_types:List[TensorType], output_types:List[TensorType],
                 input_constants:List[np.ndarray],
                 onnx_node:onnx.NodeProto) -> Node:
        return cls(name, input_nodes, output_nodes, input_types, output_types, input_constants)

    @classmethod
    def unused_onnx_inputs(cls) -> List[int]:
        return []

    def get_cuda_code(self, func_sig, node_name) -> str:
        return func_sig + " {\n    // TODO\n}\n"

    def get_constant_tensors(self, node_name) -> Dict[str, np.ndarray]:
        return {}

    def get_global_params(self, node_name) -> List[Tuple[str, str]]:
        return []
    
    def get_init_code(self, node_name) -> str:
        return ""


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