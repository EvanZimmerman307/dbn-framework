from typing import Protocol, Dict, Any
import numpy as np


"""Takes node & edge scores, returns the best sequence (Viterbi, beam search, etc.)."""
class Optimizer(Protocol):
    def decode(self, nodes, edges, **kwargs) -> list[int]:
        """
        Args:
            nodes: list of node dicts, each like
                { "id": int, "time": float, "logit": float, "embedding": np.ndarray }
            edges: list of edge dicts, each like
                { "src": int, "dst": int, "logp": float }
        Returns:
            A list of chosen node IDs (path through the graph).
        """
        ...
