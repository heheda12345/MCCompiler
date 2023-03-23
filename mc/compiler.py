from mc.graph import Graph
from mc.optim import *

def compile(model_onnx):
    graph = Graph.from_onnx(model_onnx)

    RemoveUnusedOnnxInput().apply(graph)
    print(graph)