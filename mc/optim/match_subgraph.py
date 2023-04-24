from .optimization import Optimization
from mc.graph import Graph
from mc.node import Node, IndexNode, gen_name
import mc.operators as ops
from typing import Dict, List, Type
import logging
import math

class PatternToNode(Optimization):
    pattern_name: str
    pattern: Dict[str, Node]
    ordered_pattern: List[str]
    input_idx: Dict[str, int]
    def __init__(self, pattern_name:str):
        super().__init__()
        self.pattern_name = pattern_name
        self.pattern = {}
        self.ordered_pattern = []
        self.input_idx = {}
        self.output_idx = {}

    def new_node_from_matched_nodes(self, matched_nodes: Dict[str, Node]):
        raise NotImplementedError
    
    def is_match(self, node: Node, name_in_pattern: str):
        raise NotImplementedError
    
    def add_node_to_pattern(self, node_cls: Type[Node], name: str, input_names: List[str], input_indexes=None, *args, **kwargs):
        real_name = f"{self.pattern_name}.{name}"
        real_input_names = [f"{self.pattern_name}.{name}" for name in input_names]
        for inp in real_input_names:
            assert inp in self.pattern, "nodes should be added in topological order"
        if input_indexes is None:
            input_indexes = [0] * len(input_names)
        self.pattern[real_name] = node_cls(real_name, [IndexNode(self.pattern[n], i) for n, i in zip(real_input_names, input_indexes)], [], [], [], [], *args, **kwargs)
        self.ordered_pattern.append(real_name)
        if isinstance(self.pattern[real_name], ops.Input):
            self.input_idx[real_name] = len(self.input_idx)
        if isinstance(self.pattern[real_name], ops.Output):
            self.output_idx[real_name] = len(self.output_idx)

    def match(self, graph: Graph, matched: Dict[str, Node]):
        depth = len(matched)
        if depth == len(self.ordered_pattern):
            return matched
        to_match = self.pattern[self.ordered_pattern[depth]]
        if isinstance(to_match, ops.Input) or isinstance(to_match, ops.Output):
            matched[to_match.name] = None
            full_match = self.match(graph, matched)
            if full_match is not None:
                return full_match
            matched.pop(to_match.name, None)
        else:
            possible_matches = [node for node in graph.nodes.values() if self.is_match(node, to_match.name)]
            for new_node in possible_matches:
                is_match = True
                for dst_idx, input_in_pattern in enumerate(to_match.input_nodes):
                    if isinstance(input_in_pattern.node, ops.Input):
                        continue
                    if new_node.input_nodes[dst_idx].node.name != matched[input_in_pattern.node.name].name:
                        is_match = False
                        break
                if not is_match: continue
                matched[to_match.name] = new_node
                full_match = self.match(graph, matched)
                if full_match is not None:
                    return full_match
            matched.pop(to_match.name, None)
        return None
        
    
    def apply(self, graph: Graph):
        num_input = len([node for node in self.pattern.values() if isinstance(node, ops.Input)])
        if num_input > 1:
            logging.warning("multiple inputs in pattern, please expect long execution time")
        while True:
            matched_nodes = self.match(graph, {})
            if matched_nodes is None:
                break
            else:
                new_node = self.new_node_from_matched_nodes(
                    {k[len(self.pattern_name)+1:]: v for k, v in matched_nodes.items()}
                )
                processed_node: set[str] = set()
                for p_node in self.pattern.values():
                    for i, p_inp in enumerate(p_node.input_nodes):
                        if isinstance(p_inp.node, ops.Input):
                            inp = matched_nodes[p_node.name].input_nodes[i]
                            print(p_node.name, i, p_inp.node.name, inp)
                            inp.node.remove_output_edge(inp.index, matched_nodes[p_node.name].name)
                            if inp.node.name in processed_node:
                                continue
                            edge_src = inp
                            edge_dst = IndexNode(new_node, self.input_idx[p_inp.node.name])
                            inp.node.add_output(edge_src, edge_dst)
                            new_node.set_input(edge_src, edge_dst)
                            processed_node.add(inp.node.name)
                            new_node.input_types[edge_dst.index] = edge_src.node.output_types[edge_src.index]
                for p_node in self.pattern.values():
                    if isinstance(p_node, ops.Output):
                        new_edge_src = IndexNode(new_node, self.output_idx[p_node.name])
                        p_src_index_node = p_node.input_nodes[0]
                        old_edge_src = IndexNode(matched_nodes[p_src_index_node.node.name], p_src_index_node.index)
                        for dst in old_edge_src.node.output_nodes[old_edge_src.index]:
                            new_edge_dst = IndexNode(dst.node, dst.index)
                            new_edge_src.node.add_output(new_edge_src, new_edge_dst)
                            dst.node.set_input(new_edge_src, new_edge_dst, allow_replace=True)
                            new_edge_src.node.output_types[new_edge_src.index] = dst.node.input_types[new_edge_dst.index]
                            old_edge_src.node.remove_output_edge(old_edge_src.index, dst.node.name)
            graph.clear_unused_nodes()


class MatchLayerNorm(PatternToNode):
    def __init__(self):
        super().__init__(self.__class__.__name__)
        self.add_node_to_pattern(ops.Input, "input", [])
        self.add_node_to_pattern(ops.ReduceMean, "mean.0", ["input"], axes=[-1], keepdims=True)
        self.add_node_to_pattern(ops.Sub, "sub.0", ["input", "mean.0"])
        self.add_node_to_pattern(ops.PowUni, "pow.0", ["sub.0"], value=2.0)
        self.add_node_to_pattern(ops.ReduceMean, "mean.1", ["pow.0"], axes=[-1], keepdims=True)
        self.add_node_to_pattern(ops.AddUni, "add.0", ["mean.1"], value="<placeholder>")
        self.add_node_to_pattern(ops.Sqrt, "sqrt.0", ["add.0"])
        self.add_node_to_pattern(ops.Div, "div.0", ["sub.0", "sqrt.0"])
        self.add_node_to_pattern(ops.Constant, "gamma", [], value="<placeholder>")
        self.add_node_to_pattern(ops.Mul, "mul.0", ["div.0", "gamma"])
        self.add_node_to_pattern(ops.Constant, "beta", [], value="<placeholder>")
        self.add_node_to_pattern(ops.Add, "add.1", ["mul.0", "beta"])
        self.add_node_to_pattern(ops.Output, "output", ["add.1"])


    def new_node_from_matched_nodes(self, matched_nodes: Dict[str, Node]) -> Node:
        return ops.LayerNorm(
            gen_name("LayerNorm"),
            [None],
            [[]],
            [None],
            [None],
            [],
            eps=matched_nodes["add.0"].value,
            gamma=matched_nodes["gamma"].value,
            beta=matched_nodes["beta"].value
        )
    
    def is_match(self, node: Node, name_in_pattern: str):
        return isinstance(node, self.pattern[name_in_pattern].__class__)

class MatchGELU(PatternToNode):
    def __init__(self):
        super().__init__(self.__class__.__name__)
        self.add_node_to_pattern(ops.Input, "input", [])
        self.add_node_to_pattern(ops.DivUni, "div.0", ["input"], value=math.sqrt(2))
        self.add_node_to_pattern(ops.Erf, "erf.0", ["div.0"])
        self.add_node_to_pattern(ops.AddUni, "add.0", ["erf.0"], value=1)
        self.add_node_to_pattern(ops.Mul, "mul.0", ["input", "add.0"])
        self.add_node_to_pattern(ops.MulUni, "mul.1", ["mul.0"], value=0.5)
        self.add_node_to_pattern(ops.Output, "output", ["mul.1"])
    
    def new_node_from_matched_nodes(self, matched_nodes: Dict[str, Node]):
        return ops.GELU(
            gen_name("GELU"),
            [None],
            [[]],
            [None],
            [None],
            []
        )

    def is_match(self, node: Node, name_in_pattern: str):
        return isinstance(node, self.pattern[name_in_pattern].__class__)
