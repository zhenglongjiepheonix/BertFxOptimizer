import operator
from abc import ABC, abstractmethod
from typing import Any, List, Set, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.fx import Graph, GraphModule, Node, replace_pattern
from torch.fx.passes.tools_common import legalize_graph


class GraphTransformPass(ABC):
    """
    Base class for torch.fx graph tranformation passes.
    """
    @abstractmethod
    def __call__(self, graph : GraphModule) -> GraphModule:
        raise NotImplementedError("Implement this first before use")
    

class DeadCodeEliminationPass(GraphTransformPass):
    def __call__(self, graph: GraphModule) -> GraphModule:
        g : Graph = graph.graph
        if g.eliminate_dead_code():
            graph.recompile()
            graph.graph.lint()
        return graph


class ConstantFoldPass(GraphTransformPass):
    def __call__(self, graph: GraphModule) -> GraphModule:
        # from torch.fx.experimental.const_fold import split_const_subgraphs
        # return split_const_subgraphs(graph)
        pass

class MergeLinearWithSameSource(GraphTransformPass):
    """This pass merges downstream `nn.Linear` layers which share the same
    input matrix into a big `nn.Linear` layer, then split the result matrix
    into same number of matrices correspondingly, a great example can be fusing
    the qkv linear layers in transformer-based models

      | Embedding Matrix|                       | Embedding Matrix|
     /         |         \                               |
 |Linear|   |Linear|   |Linear|     -->               |Linear|
    |          |          |                          /   |   \ 
    Q          K          V                         Q    K    V
  
    """

    def process(
        self,
        graph: GraphModule,
        upstream_node: Node,
        downstream_nodes_and_mods: List[Tuple[Node, nn.Module]],
    ):
        device, dtype, in_dims, weights = None, None, None, []
        for node, mod in downstream_nodes_and_mods:
            assert isinstance(mod, nn.Linear)
            if in_dims is None:
                in_dims = mod.in_features
            elif in_dims != mod.in_features:
                raise RuntimeError(f"found different input dims of Linear layers sharing the same input")
            
            if device is None or dtype is None:
                device = mod.weight.device
                dtype = mod.weight.dtype
            elif not(mod.weight.device == device) or not(mod.weight.dtype is dtype):
                raise RuntimeError(f"found mismatched devices or dtypes when trying to fuse {node.name}")
            weights.append(mod.weight)
        
        out_dims = [ts.shape[0] for ts in weights]
        need_bias = any([hasattr(mod, 'bias') for _,mod in downstream_nodes_and_mods])
        new_linear_layer = nn.Linear(in_features=in_dims, out_features=sum(out_dims), bias=need_bias, device=device, dtype=dtype)

        with torch.no_grad():
            new_linear_layer.weight = nn.Parameter(torch.cat(weights, dim = 0))
            new_linear_layer.weight.to(device=device, dtype=dtype)
            if need_bias:
                biases = [mod.bias if hasattr(mod, 'bias') else torch.zeros((mod.weight.shape[0],)) for _,mod in downstream_nodes_and_mods]
                new_linear_layer.bias = nn.Parameter(torch.cat(biases, dim=0))
                new_linear_layer.bias.to(device=device, dtype=dtype)

        g : Graph = graph.graph
        new_module_name = '-'.join([node.target.split('.')[-1] for node,_ in downstream_nodes_and_mods])
        new_module_name = '.'.join(downstream_nodes_and_mods[0][0].target.split('.')[:-1] + [new_module_name])
        
        # we need to modify module before updating graph structure
        graph.add_submodule(new_module_name, new_linear_layer)
        for node,_ in downstream_nodes_and_mods:
            graph.delete_submodule(node.target)
        
        merged_node = g.call_module(new_module_name, args=(upstream_node, ))
        start_idx = 0
        for idx, (node, _) in enumerate(downstream_nodes_and_mods):
            node.args = (merged_node, (..., slice(start_idx, start_idx + out_dims[idx])))
            node.op = 'call_function'
            node.target = operator.getitem
            start_idx += out_dims[idx]


    def __call__(self, graph: GraphModule) -> GraphModule:
        g : Graph = graph.graph
        patterns = {}
        for node in g.nodes:
            assert isinstance(node, Node)
            if node.op == 'call_module':
                mod = graph.get_submodule(node.target)
                if not isinstance(mod, nn.Linear) : continue
                upstream_node = node.args[0]
                if upstream_node in patterns: continue
                
                nodes_and_mods = []
                for user_node in upstream_node.users:
                    if user_node.op == 'call_module':
                        user_mod = graph.get_submodule(user_node.target)
                        if isinstance(user_mod, nn.Linear):
                            nodes_and_mods.append((user_node, user_mod))
                if len(nodes_and_mods) > 1:
                    patterns[upstream_node] = nodes_and_mods
        if not patterns: return graph

        for upstream_node, nodes_and_mods in patterns.items():
            self.process(graph, upstream_node, nodes_and_mods)

        legalize_graph(graph)
        graph.recompile()
        graph.graph.lint()
        return graph


class RemoveDropoutPass(GraphTransformPass):
    """Dropout layer can be simply removed during inference
    """
    def process(
        self,
        graph: GraphModule,
        dropout_layer: Node
    ):
        upstream_node = dropout_layer.args[0]
        dropout_layer.replace_all_uses_with(upstream_node)
        graph.delete_submodule(dropout_layer.target)
        graph.graph.erase_node(dropout_layer)
        
    def __call__(self, graph: GraphModule) -> GraphModule:
        g : Graph = graph.graph
        dropout_layers = []
        for node in g.nodes:
            assert isinstance(node, Node)
            if node.op == 'call_module':
                mod = graph.get_submodule(node.target)
                if isinstance(mod, nn.Dropout):
                    dropout_layers.append(node)
        if not dropout_layers: return graph
        for dropout_layer in dropout_layers: self.process(graph, dropout_layer)

        graph.recompile()
        graph.graph.lint()
        return graph


class FuseDivIntoQKPass(GraphTransformPass):
    """Fuse div after matmul (Q,K) into one of Q or K, this
    is a conservative pass which will only apply if matched patterns
    are well-formed as expected:

        matmul(Q,K) / sqrt(d_k) --> matmul(Q_normalized, K)
    
    """
    acceptable_relay_nodes : Set[Tuple[str, Any]] = {
        # shape-related operations which don't alter value
        ('call_method', 'size'),
        ('call_method', 'transpose'),
        ('call_method', 'permute'),
        ('call_method', 'contiguous'),
        ('call_method', 'reshape'),
        ('call_method', 'view'),
        ('call_function', torch.transpose),
        ('call_function', torch.permute),
        ('call_function', torch.reshape),
    }

    def can_fuse(self, node : Node) -> bool:
        for user in node.users:
            if not((user.op, user.target) in self.acceptable_relay_nodes):
                return False
        return True


    def process(
        self,
        graph: GraphModule,
        candidate_div_node : Node,
    ):
        matmul : Node = candidate_div_node.args[0]
        lhs, rhs = matmul.args[:2]
        d_k = candidate_div_node.args[1]

        success = False
        while (lhs.op, lhs.target) in self.acceptable_relay_nodes and \
            len(lhs.users) == 1 and len(lhs.args) and isinstance(lhs.args[0], Node):
            lhs = lhs.args[0]
        
        if lhs.op == 'call_module':
            mod = graph.get_submodule(lhs.target)
            if isinstance(mod, nn.Linear) and self.can_fuse(lhs):
                try:
                    with torch.no_grad():
                        mod.weight.true_divide_(d_k)
                        if hasattr(mod, 'bias'):
                            mod.bias.true_divide_(d_k)
                    success = True
                except Exception as e:
                    # simply fall back if any error happens
                    pass
        
        if not success:
            while (rhs.op, rhs.target) in self.acceptable_relay_nodes and \
            len(rhs.users) == 1 and len(rhs.args) and isinstance(rhs.args[0], Node):
                rhs = rhs.args[0]
            if rhs.op == 'call_module':
                mod = graph.get_submodule(rhs.target)
                if isinstance(mod, nn.Linear) and self.can_fuse(rhs):
                    try:
                        with torch.no_grad():
                            mod.weight.true_divide_(d_k)
                            if hasattr(mod, 'bias'):
                                mod.bias.true_divide_(d_k)
                        success = True
                    except:
                        # simply fall back if any error happens
                        pass
        if success:
            candidate_div_node.replace_all_uses_with(matmul)
            graph.graph.erase_node(candidate_div_node)
    

    def __call__(self, graph: GraphModule) -> GraphModule:
        g : Graph = graph.graph
        possible_candidates = []
        for node in g.nodes:
            assert isinstance(node, Node)

            if (node.op == 'call_function' and node.target is operator.truediv \
            or node.op == 'call_function' and node.target is torch.div or
            node.op == 'call_method' and node.target in {'div', 'div_'}) and \
            isinstance(node.args[0], Node) and node.args[0].op == 'call_function' \
            and len(node.args[0].users) == 1 and node.args[0].target is torch.matmul:
                possible_candidates.append(node)
        
        for candidate in possible_candidates:
            self.process(graph, candidate)

        graph.recompile()
        graph.graph.lint()

        return graph


class FuseAttentionPass(GraphTransformPass):
    """This might not generalize to other attention patterns, just a quick
    hack to bert traced from `transformers` library
    """
    @staticmethod
    def original_pattern(q : torch.Tensor, k : torch.Tensor, v : torch.Tensor, attn_mask : torch.Tensor):
        return torch.matmul(
            F.softmax(torch.matmul(q.view(q.size()[slice(None, -1, None)] + (12,64)).permute(0,2,1,3), \
                k.view(k.size()[slice(None, -1, None)] + (12, 64)).permute(0,2,1,3).transpose(-1,-2)) / 8.0 \
                + attn_mask, dim = -1, _stacklevel = 3, dtype=None), 
                v.view(v.size()[slice(None, -1, None)] + (12, 64)).permute(0,2,1,3)
            )
    @staticmethod
    def replace_pattern(q : torch.Tensor, k : torch.Tensor, v : torch.Tensor, attn_mask : torch.Tensor):
        return F.scaled_dot_product_attention(q.view(q.size()[slice(None, -1, None)] + (12,64)).permute(0,2,1,3), \
            k.view(k.size()[slice(None, -1, None)] + (12, 64)).permute(0,2,1,3), \
            v.view(v.size()[slice(None, -1, None)] + (12, 64)).permute(0,2,1,3), \
            attn_mask = attn_mask
        )
    
    def __call__(self, graph: GraphModule) -> GraphModule:
        replace_pattern(graph, self.original_pattern, self.replace_pattern)
        graph.recompile()
        graph.graph.lint()
        return graph


# class ShapeRelatedNodeAnnotationPass(GraphTransformPass):
#     """This pass tries to annotate operations relate with shape transformation,
#     since shape-related operations are always tricky and variable, this is a simple
#     pass trying to find possible shape-related nodes, may need more thoughts and refinement
#     in order to scale ...
#     """
#     @staticmethod
#     def shape_start_pattern(n : Node) -> bool:
#         if n.type == 'call_method' and n.target == 'size':
#             return True
#         if n.type == 'call_function' and n.target is torch.Size:
#             return True
        
#         # may overlook other cases which generate shapes
#         return False
    

#     @staticmethod
#     def shape_relay_pattern(from_node : Node, to_node : Node) -> bool:
#         pass
    
#     @staticmethod
#     def shape_end_pattern(n : Node) -> bool:
#         if n.type == 'call_method' and n.target == 'view':
#             return True
#         if n.type == 'call_method' and n.target == 'reshape':
#             return True
#         if n.type == 'call_function' and n.target is torch.reshape:
#             return True
#
#     def __call__(self, graph: GraphModule) -> bool: