from mc.node import Node, NodeType, IndexNode
from typing import Dict, Tuple, List
from mc.types import TensorType
import mc.node_utils as node_utils
import onnx
import numpy as np

class Graph:
    nodes: Dict[str, Node]
    inputs: List[Node]
    outputs: List[Node]

    def __init__(self) -> None:
        self.nodes = {}
        self.inputs = []
        self.outputs = []
    
    @classmethod
    def from_onnx(cls, onnx_graph: onnx.GraphProto):
        graph = cls()
        sources: Dict[str, Tuple[str, int]] = {}
        for node in onnx_graph.input:
            sources[node.name] = (node.name, 0)
        for node in onnx_graph.initializer:
            sources[node.name] = (node.name, 0)
        for node in onnx_graph.node:
            for i, output in enumerate(node.output):
                sources[output] = (node.name, i)
        
        type_of_tensor: Dict[str, TensorType] = {}
        for tensor in onnx_graph.value_info:
            type_of_tensor[tensor.name] = TensorType.from_onnx_type(tensor.type)
        for tensor in onnx_graph.input:
            type_of_tensor[tensor.name] = TensorType.from_onnx_type(tensor.type)
        for tensor in onnx_graph.initializer:
            type_of_tensor[tensor.name] = TensorType.from_onnx_tensor(tensor)
        for tensor in onnx_graph.output:
            type_of_tensor[tensor.name] = TensorType.from_onnx_type(tensor.type)
        
        initializers: Dict[str, np.ndarray] = {}
        for tensor in onnx_graph.initializer:
            initializers[tensor.name] = onnx.numpy_helper.to_array(tensor)

        for onnx_input in onnx_graph.input:
            graph.add_node(node_utils.parse_input_from_onnx(
                onnx_input, type_of_tensor[onnx_input.name]))
            graph.inputs.append(graph.nodes[onnx_input.name])
        for onnx_initializer in onnx_graph.initializer:
            graph.add_node(node_utils.parse_initializer_from_onnx(
                onnx_initializer,
                type_of_tensor[onnx_initializer.name]))
        for onnx_node in onnx_graph.node:
            node = node_utils.parse_op_from_onnx(
                onnx_node,
                [type_of_tensor[name] for name in onnx_node.input],
                [type_of_tensor[name] for name in onnx_node.output],
                [initializers[name] if name in initializers else None for name in onnx_node.input ]
            )
            graph.add_node(node)
            for dst_idx, dst_name in enumerate(onnx_node.input):
                src_name, src_idx = sources[dst_name]
                graph.add_edge(
                    IndexNode(graph.nodes[src_name], src_idx),
                    IndexNode(node, dst_idx)
                )
        for onnx_output in onnx_graph.output:
            node = node_utils.parse_output_from_onnx(
                onnx_output, type_of_tensor[onnx_output.name])
            graph.add_node(node)
            src_name, src_idx = sources[onnx_output.name]
            graph.add_edge(
                IndexNode(graph.nodes[src_name], src_idx),
                IndexNode(node, 0)
            )
            graph.outputs.append(node)

        for node in graph.nodes.values():
            assert node.input_exist(), f'Node {node.name} has None in input'
        return graph


    def add_node(self, node: Node):
        assert node.name not in self.nodes, f'Node {node.name} already exists'
        self.nodes[node.name] = node
    

    def add_edge(self, src: IndexNode, dst: IndexNode):
        src.node.add_output(src, dst)
        dst.node.set_input(src, dst)
    

    def get_node(self, name: str) -> Node:
        return self.nodes[name]
    

    def fill_out_edges(self):
        for node in self.nodes.values():
            for i, input in enumerate(node.inputs):
                if input is not None:
                    input.node.add_output(input, IndexNode(node, i))
    
    def __str__(self) -> str:
        return '\n'.join([str(node) for node in self.nodes.values()])

    
