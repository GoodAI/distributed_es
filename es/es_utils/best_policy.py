from typing import Optional

from es.es_utils.es_common import SharedNoiseTable, Result
from policy.policy import Policy


def _update(fitness: float,
            noise_idx: int,
            sign: int,
            best_policy: Policy,
            policy: Policy,
            best_f: Optional[float],
            noise: SharedNoiseTable,
            sigma: float) -> Optional[float]:
    """If better than currently best fitness found:
     decode deltas, apply to the current genome, store in best_policy"""
    if best_f is None or fitness > best_f:
        print(f'HEAD: Better fitness found, {fitness} > {best_f}, updating best_policy')
        current_genome = policy.serialize_to_genome()
        deltas = noise.get(noise_idx, policy.num_params)
        best_genome = current_genome + sign * sigma * deltas
        best_policy.deserialize_from_genome(best_genome)
        return fitness

    return best_f


def _update_best_policy(best_policy: Policy,
                        policy: Policy,
                        best_f: Optional[float],
                        results: Result,
                        noise: SharedNoiseTable,
                        sigma: float) -> Optional[float]:
    """Keep track of the best individual, update the best policy (genome) and return best f"""

    for two_returns, noise_idx in zip(results.returns_n2, results.noise_inds_n):
        best_f = _update(two_returns[0], noise_idx, +1, best_policy, policy, best_f, noise, sigma)
        best_f = _update(two_returns[1], noise_idx, -1, best_policy, policy, best_f, noise, sigma)

    return best_f
