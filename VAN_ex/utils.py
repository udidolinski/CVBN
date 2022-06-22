from __future__ import annotations
import heapq
from typing import List, Dict, Any, Tuple
import numpy as np
import gtsam
from numpy.typing import NDArray

FloatNDArray = NDArray[np.float64]


class PriorityQueue:
    def __init__(self):
        self.heap = []
        self.init = False

    def push(self, item: Node, priority: np.float, cov_sum: gtsam.noiseModel.Gaussian.Covariance) -> None:
        if not self.init:
            self.init = True
            try:
                item < item
            except Exception:
                if cov_sum is None:
                    self.init = False
                else:
                    cov_sum.__class__.__lt__ = lambda x, y: (True)
                    item.__class__.__lt__ = lambda x, y: (True)
        pair = (priority, item, cov_sum)
        heapq.heappush(self.heap, pair)

    def pop(self) -> Tuple[Node, np.float, gtsam.noiseModel.Gaussian.Covariance]:
        (priority, item, cov_sum) = heapq.heappop(self.heap)
        return item, priority, cov_sum

    def isEmpty(self) -> bool:
        return len(self.heap) == 0


class Node:

    def __init__(self, symbol: gtsam.symbol, neighbors_cov_dict: Dict[Node, gtsam.noiseModel.Gaussian.Covariance]):
        self.__symbol = symbol
        self.__neighbors_cov_dict = neighbors_cov_dict

    def __eq__(self, other: Node):
        return self.__symbol == other.__symbol

    def __hash__(self):
        return hash(self.__symbol)

    def __str__(self):
        return str(self.__symbol)

    def add_neighbor(self, neighbor_node: Node, relative_cov: gtsam.noiseModel.Gaussian.Covariance) -> None:
        self.__neighbors_cov_dict[neighbor_node] = relative_cov

    def get_edge_cov(self, neighbor_node: Node) -> gtsam.noiseModel.Gaussian.Covariance:
        return self.__neighbors_cov_dict[neighbor_node]

    def get_symbol(self) -> gtsam.symbol:
        return self.__symbol

    def get_neighbors_cov_dict(self) -> Dict[Node, gtsam.noiseModel.Gaussian.Covariance]:
        return self.__neighbors_cov_dict

    def set_neighbors_cov_dict(self, d: Dict[Node, gtsam.noiseModel.Gaussian.Covariance]) -> None:
        self.__neighbors_cov_dict = d


def get_weight(current_node, neighbor) -> np.float:
    covariance = current_node.get_edge_cov(neighbor).covariance()
    return np.linalg.det(covariance)


def search(s_node: Node, t_node: Node) -> Tuple[gtsam.noiseModel.Gaussian.Covariance, bool]:
    visited_nodes = set()
    min_heap = PriorityQueue()
    min_heap.push(s_node, 0, None)
    while not min_heap.isEmpty():
        current_node, curr_priority, current_cov_sum = min_heap.pop()
        if current_node == t_node:
            return current_cov_sum, True
        elif current_node not in visited_nodes:
            for neighbor, covariance in current_node.get_neighbors_cov_dict().items():
                edge_weight = get_weight(current_node, neighbor)
                cov_sum = current_cov_sum.covariance() + covariance.covariance() if current_cov_sum is not None else covariance.covariance()
                min_heap.push(neighbor, curr_priority+edge_weight, gtsam.noiseModel.Gaussian.Covariance(cov_sum))
        visited_nodes.add(current_node)
    return np.zeros((6, 6)), False

# e = Node(1, {})
# d = Node(2, {})
# c.txt = Node(3, {})
# b = Node(4, {})
# a = Node(5, {})
#
# a.set_neighbors_cov_dict({c.txt: "c.txt", b: "b"})
# b.set_neighbors_cov_dict({c.txt: "c.txt", a: "a", d: "d"})
# c.txt.set_neighbors_cov_dict({b: "b", a: "a", d: "d"})
# d.set_neighbors_cov_dict({c.txt: "c.txt", b: "b", e: "e"})
# e.set_neighbors_cov_dict({d: "d"})
#
#
# res = search(a, b)
# print(res)
