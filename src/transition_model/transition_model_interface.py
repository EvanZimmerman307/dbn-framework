from typing import Protocol, Dict, Any
import numpy as np

"""Each edge between two candidates is scored."""

class TransitionModel(Protocol):
    def log_prob(self, prev_time: float, curr_time: float,
                 prev_embedding: np.ndarray | None = None,
                 curr_embedding: np.ndarray | None = None,
                 **kwargs) -> float:
        """
        Args:
            prev_time: time (s) of previous candidate
            curr_time: time (s) of current candidate
            prev_embedding, curr_embedding: optional embeddings for conditioning
        Returns:
            log probability of transition (prev → curr)
        """
        ...
    
    def fit(self, sequences: list[list[float]], **kwargs) -> None:
        """Fit transition parameters (e.g., EM for GMM on RR intervals)."""
        ...
