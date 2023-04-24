from mc.graph import Graph
from mc.optim import *
import logging

def compile(model_onnx):
    graph = Graph.from_onnx(model_onnx)

    RemoveUnusedOnnxInput().apply(graph)
    RemoveViewOp().apply(graph)
    MatchLayerNorm().apply(graph)
    MatchGELU().apply(graph)
    MatchCublasEPILOGUE().apply(graph)
    MatchCublasPROLOGUE().apply(graph)
    logging.info("after compile\n" + graph.str_in_topological_order())
