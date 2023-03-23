from .optimization import Optimization
from mc.graph import Graph

class RemoveUnusedOnnxInput(Optimization):
    def apply(self, graph: Graph):
        for node in graph.nodes.values():
            for inp in node.unused_onnx_inputs():
                graph.remove_node_input_and_edge(node, inp)
        graph.clear_unused_nodes()