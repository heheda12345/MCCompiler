from mc.node import Node, NodeType, IndexNode
from typing import Dict, Tuple, List
from mc.types import TensorType
import mc.node_utils as node_utils
import onnx
import numpy as np
import logging

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
    
    def remove_node_input_and_edge(self, node: Node, index: int):
        node.input_nodes[index].node.remove_output_edge(node.input_nodes[index].index, node.name)
        node.remove_input(index)
    
    def remove_node(self, node: Node, check=True):
        if check:
            assert len(node.input_nodes) == 1
            assert len(node.output_nodes) == 1
            assert node.input_types[0].size() == node.output_types[0].size()
        else:
            logging.warning(f'Node {node.name} is removed without checking')
        src_node = node.input_nodes[0].node
        dst_nodes = node.output_nodes[0]
        src_node.remove_output_edge(node.input_nodes[0].index, node.name)
        for dst_node in dst_nodes:
            src_node.add_output(node.input_nodes[0], dst_node)
            dst_node.node.input_nodes[dst_node.index] = IndexNode(src_node, node.input_nodes[0].index)

    def get_node(self, name: str) -> Node:
        return self.nodes[name]

    def clear_unused_nodes(self):
        used_nodes = {}
        # bfs from the output nodes
        queue = self.outputs.copy()
        while len(queue) > 0:
            node = queue.pop()
            used_nodes[node.name] = node
            for input in node.input_nodes:
                assert input is not None
                if input.node not in used_nodes:
                    queue.append(input.node)
        for node in self.inputs:
            used_nodes[node.name] = node

        # print unused nodes
        for node in self.nodes.values():
            if node.name not in used_nodes:
                logging.info(f'remove unused node: {node.name}')
        self.nodes = used_nodes

        # filter unused edges
        for node in self.nodes.values():
            for i, output_node in enumerate(node.output_nodes):
                output_node = [o for o in output_node if o.node.name in used_nodes]
                node.output_nodes[i] = output_node

    def fill_out_edges(self):
        for node in self.nodes.values():
            for i, input in enumerate(node.inputs):
                if input is not None:
                    input.node.add_output(input, IndexNode(node, i))
    
    def topological_sort(self) -> List[Node]:
        # topolic sort
        queue: List[Node] = []
        for node in self.nodes.values():
            if len(node.input_nodes) == 0:
                queue.append(node)
        remain_degree = {}
        for node in self.nodes.values():
            remain_degree[node.name] = len(node.input_nodes)
        # for node in self.nodes.values():
        #     print(node)
        # print("before sort", remain_degree['Transpose_49'], remain_degree['onnx::Gemm_102'])
        sorted_nodes = []
        while len(queue) > 0:
            node = queue.pop()
            sorted_nodes.append(node)
            # print("before processing", node.name, remain_degree['Transpose_49'], remain_degree['onnx::Gemm_102'])
            for output in node.output_nodes:
                for o in output:
                    # if node.name == 'Transpose_49': print(o.node.name, remain_degree[o.node.name])
                    remain_degree[o.node.name] -= 1
                    if remain_degree[o.node.name] == 0:
                        queue.append(o.node)
        if len(sorted_nodes) != len(self.nodes):
            # print(len(sorted_nodes), len(self.nodes))
            sorted_node_names = [node.name for node in sorted_nodes]
            for node in self.nodes.values():
                if node.name not in sorted_node_names:
                    logging.error(f'mismatch node: {node.name}, remain_degree = {remain_degree[node.name]}')
            raise RuntimeError('Graph is not a DAG')
        return sorted_nodes
    
    def str_in_topological_order(self):
        sorted_nodes = self.topological_sort()
        return '\n'.join([str(node) for node in sorted_nodes])
    
    def __str__(self) -> str:
        return '\n'.join([str(node) for node in self.nodes.values()])


    
