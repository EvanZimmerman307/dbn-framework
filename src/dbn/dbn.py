from ..emission_model.emission_model_interface import EmissionModel
from ..transition_model.transition_model_interface import TransitionModel
from ..optimizer.optimizer_interface import Optimizer
import numpy as np

class DBN:
    def __init__(self, emission: EmissionModel,
                       transition: TransitionModel,
                       optimizer: Optimizer):
        self.emission = emission
        self.transition = transition
        self.optimizer = optimizer

    def run(self, candidates: list[float], snippets: dict[float, np.ndarray]) -> list[float]:
        nodes, edges = [], []

        # Score nodes
        for i, t in enumerate(candidates):
            out = self.emission.score(snippets[t])
            nodes.append({"id": i, "time": t, **out})

        # Score edges
        for j in nodes:
            for i in nodes:
                if i["time"] > j["time"]:
                    logp = self.transition.log_prob(j["time"], i["time"],
                                                    j.get("embedding"), i.get("embedding"))
                    edges.append({"src": j["id"], "dst": i["id"], "logp": logp})

        # Decode path
        best_path = self.optimizer.decode(nodes, edges)
        return [nodes[i]["time"] for i in best_path]
