from typing import Callable, List, Set

from torch.fx import Graph, Node


class GraphSearchEngine:
    """Graph search engine class enables vague pattern search, currently
    it supports returning a set of nodes regarding to the given search conditons.
    See `opset_matching` for details.
    """

    def __init__(self, graph : Graph) -> None:
        self._graph = graph
        self._cache = {}

    @property
    def graph(self) -> Graph:
        return self._graph
    
    @staticmethod
    def get_upstream_ops(n : Node) -> List[Node]:
        return n.all_input_nodes
    
    @staticmethod
    def get_downstream_ops(n : Node) -> List[Node]:
        return [node for node in n.users]

    def _opset_matching(
        self,
        start_point : Node,
        relay_expr : Callable,
        end_expr : Callable,
        direction : str
    ) -> Set[Node]:
        if start_point in self._cache : return self._cache[start_point]

        ret_collection = set()
        
        # a bottom-up search
        if direction ==  'up': next_ops = GraphSearchEngine.get_upstream_ops(start_point)
        else : next_ops = GraphSearchEngine.get_downstream_ops(start_point)
        
        for op in next_ops:
            # find a valid end point
            if end_expr is not None and end_expr(op):
                ret_collection.update([start_point, op])
            else:
                # if operation is not a valid relay operation
                if relay_expr is not None and not relay_expr(start_point, op):
                    continue

                next_op_set = self._opset_matching(
                    start_point = op,
                    relay_expr = relay_expr,
                    end_expr = end_expr,
                    direction = direction
                )
                
                if len(next_op_set) > 0:
                    ret_collection.update(next_op_set)

        self._cache[start_point] = ret_collection 
        return ret_collection


    def opset_matching(
        self,
        start_expr : Callable = lambda x : True,
        relay_expr : Callable = lambda x, y : True,
        end_expr : Callable = lambda x : True,
        direction : str = 'down'
    ) -> Set[Node]:
        """Matches a set of nodes which are either end points or relay points on pathes
        which statisfy the given `start_expr` --> `relay_expr` --> `end_expr` pattern

        Args:
            start_expr (Callable): starting node condition
            relay_expr (Callable): relay node condition
            end_expr (Callable): end node condition
            direction (str): `up` or `down`

        Return:
            A set of all matched nodes
        """
        
        ret_collection, starting_candidates = set(), set()
        for node in self.graph.nodes:
            if start_expr(node): starting_candidates.add(node)
        
        for op in starting_candidates:
            partial_matchings = self._opset_matching(
                start_point = op, relay_expr = relay_expr,
                end_expr = end_expr, direction = direction
            )
            ret_collection.add(partial_matchings)
        
        self._cache.clear()
        return ret_collection
