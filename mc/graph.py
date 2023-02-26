from mc.node import Node, NodeType, IndexNode
from typing import Dict, Tuple
import onnx

class Graph:
    nodes: Dict[str, Node]

    def __init__(self) -> None:
        self.nodes = {}
    
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

        # add nodes to graph
        for onnx_input in onnx_graph.input:
            graph.add_node(Node.from_onnx(onnx_input, NodeType.input))
        for onnx_initializer in onnx_graph.initializer:
            graph.add_node(Node.from_onnx(onnx_initializer, NodeType.initializer))
        for onnx_node in onnx_graph.node:
            node = Node.from_onnx(onnx_node, NodeType.node)
            graph.add_node(node)
            for dst_idx, dst_name in enumerate(onnx_node.input):
                src_name, src_idx = sources[dst_name]
                graph.add_edge(
                    IndexNode(graph.nodes[src_name], src_idx),
                    IndexNode(node, dst_idx)
                )        
        for onnx_output in onnx_graph.output:
            node = Node.from_onnx(onnx_output, NodeType.output)
            graph.add_node(node)
            src_name, src_idx = sources[onnx_output.name]
            graph.add_edge(
                IndexNode(graph.nodes[src_name], src_idx),
                IndexNode(node, 0)
            )

        for node in graph.nodes.values():
            assert node.io_all_exist(), f'Node {node.name} has None in input/output'

        return graph


    def add_node(self, node: Node):
        assert node.name not in self.nodes, f'Node {node.name} already exists'
        self.nodes[node.name] = node
    

    def add_edge(self, src: IndexNode, dst: IndexNode):
        print("add_edge:", src, dst)
        src.node.set_output(src, dst)
        dst.node.set_input(src, dst)
    
    def __str__(self) -> str:
        print("str called")
        # concat the str(node) with '\n'
        return '\n'.join([str(node) for node in self.nodes.values()])

    
