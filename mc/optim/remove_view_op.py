from .optimization import Optimization
from mc.graph import Graph
import mc.operators as ops
from typing import List

class RemoveViewOp(Optimization):
    def __init__(self) -> None:
        super().__init__()

    def apply(self, graph: Graph):
        removed_node_names: List[str] = []
        for node in graph.nodes.values():
            if isinstance(node, ops.Reshape):
                assert len(node.output_nodes) == 1
                for output in node.output_nodes[0]:
                    output.node.input_types[output.index] = node.input_types[0]
                graph.remove_node(node)
                removed_node_names.append(node.name)
        for name in removed_node_names:
            graph.nodes.pop(name)
