from .match_subgraph import PatternToNode
from .optimization import Optimization
from mc.graph import Graph
from mc.node import Node
import mc.operators as ops
from typing import Dict


class Match_CUBLASLT_EPILOGUE_GELU_BIAS(PatternToNode):
    def __init__(self):
        super().__init__(self.__class__.__name__)
        self.add_node_to_pattern(ops.Input, "input.0", [])
        self.add_node_to_pattern(ops.Input, "input.1", [])
        self.add_node_to_pattern(ops.Input, "input.bias", [])
        self.add_node_to_pattern(ops.UniMatMul, "matmul.0", ["input.0", "input.1"], )
        self.add_node_to_pattern(ops.Add, "add.0", ["input.bias", "matmul.0"])
        self.add_node_to_pattern(ops.GELU, "gelu.0", ["add.0"])
        self.add_node_to_pattern(ops.Output, "output", ["gelu.0"])
    
    def new_node_from_matched_nodes(self, matched_nodes: Dict[str, Node]):
        new_node = ops.UniMatMul.copy_attr_from(matched_nodes["matmul.0"], num_inputs=3)
        new_node.epilogue = ops.matmul.Epilogue.CUBLASLT_EPILOGUE_GELU
        return new_node
    
    def is_match(self, node: Node, name_in_pattern: str):
        if name_in_pattern == "matmul.0": return len(node.input_types) == 2
        return isinstance(node, self.pattern[name_in_pattern].__class__)

class Match_CUBLASLT_EPILOGUE_BIAS(PatternToNode):
    def __init__(self):
        super().__init__(self.__class__.__name__)
        self.add_node_to_pattern(ops.Input, "input.0", [])
        self.add_node_to_pattern(ops.Input, "input.1", [])
        self.add_node_to_pattern(ops.Input, "input.bias", [])
        self.add_node_to_pattern(ops.UniMatMul, "matmul.0", ["input.0", "input.1"], )
        self.add_node_to_pattern(ops.Add, "add.0", ["input.bias", "matmul.0"])
        self.add_node_to_pattern(ops.Output, "output", ["add.0"])
    
    def new_node_from_matched_nodes(self, matched_nodes: Dict[str, Node]):
        new_node = ops.UniMatMul.copy_attr_from(matched_nodes["matmul.0"], num_inputs=3)
        return new_node
    
    def is_match(self, node: Node, name_in_pattern: str):
        if name_in_pattern == "matmul.0": return len(node.input_types) == 2
        return isinstance(node, self.pattern[name_in_pattern].__class__)

class Match_CUBLASLT_EPILOGUE_BIAS_V2(PatternToNode):
    def __init__(self):
        super().__init__(self.__class__.__name__)
        self.add_node_to_pattern(ops.Input, "input.0", [])
        self.add_node_to_pattern(ops.Input, "input.1", [])
        self.add_node_to_pattern(ops.Input, "input.bias", [])
        self.add_node_to_pattern(ops.UniMatMul, "matmul.0", ["input.0", "input.1"], )
        self.add_node_to_pattern(ops.Add, "add.0", ["matmul.0", "input.bias"])
        self.add_node_to_pattern(ops.Output, "output", ["add.0"])
    
    def new_node_from_matched_nodes(self, matched_nodes: Dict[str, Node]):
        new_node = ops.UniMatMul.copy_attr_from(matched_nodes["matmul.0"], num_inputs=3)
        return new_node
    
    def is_match(self, node: Node, name_in_pattern: str):
        print("name_in_pattern", name_in_pattern)
        if name_in_pattern == f"{self.__class__.__name__}.matmul.0":
            print("node:", node)
            return len(node.input_nodes) == 2
        return isinstance(node, self.pattern[name_in_pattern].__class__)


class MatchCublasEPILOGUE(Optimization):
    def __init__(self):
        super().__init__()
    
    def apply(self, graph: Graph):
        # from complex patterns to simple patterns
        pattern_to_node = [
            Match_CUBLASLT_EPILOGUE_GELU_BIAS(),
            Match_CUBLASLT_EPILOGUE_BIAS(),
            Match_CUBLASLT_EPILOGUE_BIAS_V2(),
        ]
        for ptn in pattern_to_node:
            ptn.apply(graph)
