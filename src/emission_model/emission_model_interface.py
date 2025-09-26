from typing import Protocol, Dict, Any
import numpy as np

"""
The purpose of the Emission Model is to capture P(E|h)
"""

class EmissionModel(Protocol):
    def score(self, snippet: np.ndarray, **kwargs) -> Dict[str, Any]:
        """
        Args:
            snippet: np.ndarray of shape (C, T) or (T,) — window of data around candidate.
        Returns:
            {
              "logits": list(float),           # class logits
              "probs": list(float),            # calibrated P(state|snippet) if available
              "pred_clas": int,                 # class with the highest prob 
              "embedding": np.ndarray   # optional feature vector for richer DBN variants
            }
        """
        ...
    
    def train(self, dataset: Any, **kwargs) -> None:
        """Optional: fit parameters using training data."""
        ...
