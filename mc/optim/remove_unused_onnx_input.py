from .optimization import Optimization
from mc.graph import Graph

class RemoveUnusedOnnxInput(Optimization):
    def apply(self, graph: Graph):
        for node in graph.nodes.values():
            if len(node.unused_onnx_inputs()) > 0:
                unused_inputs = node.unused_onnx_inputs()
                for inp in unused_inputs:
                    graph.remove_node_input_and_edge(node, inp)
                node.input_types = [node.input_types[i] for i in range(len(node.input_types)) if i not in unused_inputs]
        graph.clear_unused_nodes()