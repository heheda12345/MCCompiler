from .match_subgraph import PatternToNode
from .optimization import Optimization
from mc.graph import Graph
from mc.node import Node
import mc.operators as ops
from typing import List
import logging
import numpy as np

class MergeLinearELmenentWise(Optimization):
    # b = a / 8
    # c = alpha * matmul(b, w) + bias
    # => c = alpha / 8 * matmul(a, w) + bias
    def apply(self, graph: Graph):
        for node in graph.nodes.values():
            if isinstance(node, ops.DivUni) \
                and len(node.output_nodes[0]) == 1 \
                and isinstance(node.output_nodes[0][0].node, ops.UniMatMul):
                input_idx = node.output_nodes[0][0].index
                matmul_node = node.output_nodes[0][0].node
                if input_idx == 0 or input_idx == 1:
                    if isinstance(node, ops.DivUni):
                        matmul_node.alpha = matmul_node.alpha / node.value.item()
                        logging.info(f"{matmul_node.name}: change alpha to {matmul_node.alpha} ({node})")
                    else:
                        raise NotImplementedError
                    # TODO: add more cases
                    graph.remove_node(node)
                else:
                    continue
        graph.clear_unused_nodes()


def perm_to_stride(perm, shape: List[int]):
    stride = [1]
    for i in range(len(shape) - 1, 0, -1):
        stride.append(stride[-1] * shape[i])
    stride.reverse()
    new_stride = []
    for x in perm:
        if x == -1:
            new_stride.append(0)
        else:
            new_stride.append(stride[x])
    return new_stride


def stride_to_perm(stride: List[int], shape: List[int]):
    assert len(stride) == len(shape)
    shape_stride = [1]
    for i in range(len(shape) - 1, 0, -1):
        shape_stride.append(shape_stride[-1] * shape[i])
    shape_stride.reverse()
    perm = []
    for s in stride:
        if s == 0:
            perm.append(-1)
        elif s in shape_stride:
            perm.append(shape_stride.index(s))
        else:
            raise ValueError("stride not match shape")
    return perm


class MergeTranspose(Optimization):
    def apply(self, graph: Graph):
        for node in graph.nodes.values():
            if isinstance(node, ops.Transpose) \
                and len(node.output_nodes[0]) == 1 \
                and isinstance(node.output_nodes[0][0].node, ops.UniMatMul):
                input_idx = node.output_nodes[0][0].index
                matmul_node = node.output_nodes[0][0].node
                if input_idx == 0 or input_idx == 1:
                    old_stride = matmul_node.input_stride[input_idx]
                    old_perm = stride_to_perm(matmul_node.input_stride[input_idx], matmul_node.input_types[input_idx].shape)
                    perm_before_transpose = []
                    for x in old_perm:
                        if x == -1:
                            perm_before_transpose.append(-1)
                        else:
                            perm_before_transpose.append(node.perm[x])
                    new_stride = perm_to_stride(perm_before_transpose, node.input_types[0].shape)
                    matmul_node.input_stride[input_idx] = new_stride
                    matmul_node.input_types[input_idx] = node.input_types[0]
                    logging.info(f"{matmul_node.name}: change input{input_idx}_stride from {old_stride} to {new_stride} due to transpose perm {node.perm}")
                    graph.remove_node(node)
                else:
                    continue
        graph.clear_unused_nodes()

class MergeSlice(Optimization):
    def apply(self, graph):
        for node in graph.nodes.values():
            if isinstance(node, ops.Slice) \
                and len(node.output_nodes[0]) == 1 \
                and isinstance(node.output_nodes[0][0].node, ops.UniMatMul):
                if len(node.axes) != 1: continue
                if node.steps[0] != 1: raise NotImplementedError
                slice_node = node
                input_idx = slice_node.output_nodes[0][0].index
                matmul_node: ops.UniMatMul = slice_node.output_nodes[0][0].node
                if input_idx == 0 or input_idx == 1:
                    old_stride = matmul_node.input_stride[input_idx]
                    perm = stride_to_perm(matmul_node.input_stride[input_idx], matmul_node.input_types[input_idx].shape)
                    new_stride = perm_to_stride(perm, slice_node.input_types[0].shape)
                    offset = np.prod(np.array(slice_node.input_types[0].shape[slice_node.axes[0] + 1:], dtype=np.int64)) * slice_node.starts[0]
                    matmul_node.input_stride[input_idx] = new_stride
                    matmul_node.input_types[input_idx] = slice_node.input_types[0]
                    matmul_node.input_offset[input_idx] += offset
                    logging.info(f"{matmul_node.name}: change input{input_idx} stride from {old_stride} to {new_stride} & add {offset} to offset due to slice {slice_node.starts[0]}:{slice_node.ends[0]}:{slice_node.steps[0]}")
                    graph.remove_node(slice_node, check=False)
                else:
                    continue
        graph.clear_unused_nodes()

class MatchCublasPROLOGUE(Optimization):
    def __init__(self):
        super().__init__()
    
    def apply(self, graph: Graph):
        print("MatchCublasPROLOGUE")
        # from complex patterns to simple patterns
        passes = [
            MergeLinearELmenentWise(),
            MergeTranspose(),
            MergeSlice(),
        ]
        for ptn in passes:
            ptn.apply(graph)