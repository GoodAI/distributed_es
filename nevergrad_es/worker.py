from argparse import Namespace
from typing import List, Tuple, Optional

import numpy as np
import ray

from utils.eval import eval_genome


def update_best(best: Optional[float], current_ones: List[float]) -> float:
    if best is None:
        return max(current_ones)
    return max(max(current_ones), best)


@ray.remote
class RayNevergradWorker:

    last_fitness: float  # last fitness collected
    last_num_steps_used: int  # how many environments steps it took

    def __init__(self):
        self.last_fitness = 0

    def submit_genome(self, conf: Namespace, genome: np.array, seed: int):
        self.last_fitness, self.last_num_steps_used = eval_genome(conf, genome, seed)

    def collect_fitness(self) -> Tuple[float, float]:
        return self.last_fitness, self.last_num_steps_used

