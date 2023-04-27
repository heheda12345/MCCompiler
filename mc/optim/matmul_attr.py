from .match_subgraph import PatternToNode
from .optimization import Optimization
from mc.graph import Graph
from mc.node import Node
import mc.operators as ops
from typing import List
import logging
import numpy as np
import copy

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


def shape_to_stride(shape: List[int]):
    stride = [1]
    for i in range(len(shape) - 1, 0, -1):
        stride.append(stride[-1] * shape[i])
    stride.reverse()
    return stride

class MergeTranspose(Optimization):
    def apply(self, graph: Graph):
        for node in graph.nodes.values():
            if isinstance(node, ops.Transpose) \
                and len(node.output_nodes[0]) == 1 \
                and isinstance(node.output_nodes[0][0].node, ops.UniMatMul):
                input_idx = node.output_nodes[0][0].index
                matmul_node = node.output_nodes[0][0].node
                if input_idx == 0 or input_idx == 1:
                    old_perm = matmul_node.input_perm[input_idx]
                    if old_perm != (0, 1, 2): # TODO: fuse multiple transposes
                        continue
                    if matmul_node.real_input_shape[input_idx] != matmul_node.input_types[input_idx].shape:
                        logging.warning(f"{matmul_node.name}: shape mismatch matmul.real {matmul_node.real_input_shape[input_idx]} vs type in graph {matmul_node.input_types[input_idx].shape}")
                        continue
                    new_perm = tuple(copy.deepcopy(node.perm))
                    matmul_node.input_perm[input_idx] = new_perm
                    matmul_node.input_types[input_idx] = node.input_types[0]
                    logging.info(f"{matmul_node.name}: change input{input_idx}_perm from {old_perm} to {new_perm} due to transpose perm {node.perm}")
                    graph.remove_node(node)
                else:
                    continue
        graph.clear_unused_nodes()


class MergeTransposeToOutput(Optimization):
    def apply(self, graph: Graph):
        for matmul_node in graph.nodes.values():
            if isinstance(matmul_node, ops.UniMatMul) \
                and isinstance(matmul_node.output_nodes[0][0].node, ops.Transpose):
                transpose_node = matmul_node.output_nodes[0][0].node
                old_perm = matmul_node.output_perm
                if old_perm != (0, 1, 2): # TODO: fuse multiple transposes
                    continue
                new_perm = tuple(copy.deepcopy(transpose_node.perm))
                matmul_node.output_perm = new_perm
                matmul_node.output_types[0] = transpose_node.output_types[0]
                logging.info(f"{matmul_node.name}: change output_perm from {old_perm} to {new_perm} due to transpose perm {transpose_node.perm}")
                graph.remove_node(transpose_node)
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
                    perm = matmul_node.input_perm[input_idx]
                    shape_after_slice = [matmul_node.real_input_shape[input_idx][perm.index(i)] for i in range(len(perm))]
                    slice_input_shape = slice_node.input_types[0].shape
                    slice_full_size = np.prod(np.array([slice_input_shape[slice_node.axes[0]:]], dtype=np.int64))
                    slice_start = np.prod(np.array(slice_node.input_types[0].shape[slice_node.axes[0] + 1:], dtype=np.int64)) * slice_node.starts[0]
                    slice_end = np.prod(np.array(slice_node.input_types[0].shape[slice_node.axes[0] + 1:], dtype=np.int64)) * slice_node.ends[0]
                    slice_size = slice_end - slice_start
                    stride_after_slice = shape_to_stride(shape_after_slice)
                    if slice_size not in stride_after_slice: continue
                    real_slice_axis = stride_after_slice.index(slice_size)
                    old_axis_len = matmul_node.real_input_shape[input_idx][real_slice_axis] 
                    new_axis_len = old_axis_len * slice_full_size // slice_size
                    matmul_node.real_input_shape[input_idx][real_slice_axis] = new_axis_len
                    matmul_node.input_types[input_idx] = slice_node.input_types[0]
                    matmul_node.input_offset[input_idx] += slice_start
                    logging.info(f"{matmul_node.name}: change input{input_idx}: add {slice_start} to offset due to slice {slice_node.starts[0]}:{slice_node.ends[0]}:{slice_node.steps[0]}")
                    graph.remove_node(slice_node, check=False)
                else:
                    continue
        graph.clear_unused_nodes()


class BMM2GEMM(Optimization):
    def apply(self, graph):
        for node in graph.nodes.values():
            if isinstance(node, ops.UniMatMul):
                print("bmm2gemm", node.name, node.size_ba, node.size_bb, node.size_m, node.size_n, node.size_k, node.input_types)
                print(node.size_ba > 1 , node.size_bb == 1 , (len(node.input_types) < 3 or node.input_types[2].size() // node.size_n == 1) , node.size_ba == node.real_input_shape[0][0] , node.size_m == node.real_input_shape[0][1])
                if node.size_ba > 1 and node.size_bb == 1 and (len(node.input_types) < 3 or node.input_types[2].size() // node.size_n == 1) and node.size_ba == node.real_input_shape[0][0] and node.size_m == node.real_input_shape[0][1]:
                    logging.info(f"{node.name}: change to bmm")
                    batch_size = node.size_ba
                    node.size_ba = 1
                    node.size_m *= batch_size
                    node.real_input_shape[0][1] *= node.real_input_shape[0][0]
                    node.real_input_shape[0][0] = 1



class MatchCublasAttrs(Optimization):
    def __init__(self):
        super().__init__()
    
    def apply(self, graph: Graph):
        # from complex patterns to simple patterns
        passes = [
            MergeLinearELmenentWise(),
            MergeTranspose(),
            MergeSlice(),
            MergeTransposeToOutput(),
            BMM2GEMM(),
        ]
        for ptn in passes:
            ptn.apply(graph)