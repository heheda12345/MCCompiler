from mc.graph import Graph

def compile(model_onnx):
    graph = Graph.from_onnx(model_onnx)
    print(graph)