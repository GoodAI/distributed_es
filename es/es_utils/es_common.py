from typing import NamedTuple

import numpy as np

LOCAL_MASTER_SOCKET_PATH = '/tmp/es_redis_master.sock'
LOCAL_RELAY_SOCKET_PATH = '/tmp/es_redis_relay.sock'
REMOTE_MASTER_SOCKET_PATH = '/var/run/redis/redis.sock'


TaskData = NamedTuple('TaskData', [
    ('genome', np.ndarray),
    ('env_seed', int)
])


TaskId = NamedTuple('TaskId', [
    ('start_time', float),
    ('generation', int),
])


Result = NamedTuple('Result',
                    [
                        ('worker_id', int),  # unique ID of worker (counting num workers)
                        ('noise_inds_n', np.ndarray),  # [N] noise seeds used for mutating?
                        ('returns_n2', np.ndarray),  # [N, 2] - rewards (mirroring theta+delta, theta-delta)
                        ('lengths_n2', np.ndarray),  # [N, 2] - steps per mutation (mirroring)
                        ('time_to_obtain', float),
                        ('time_to_eval', float)
                    ])


class SharedNoiseTable(object):
    def __init__(self):
        import ctypes, multiprocessing
        seed = 123
        count = 250000000  # 1 gigabyte of 32-bit numbers. Will actually sample 2 gigabytes below.
        print(f'Sampling {count} random numbers with seed {seed}')
        self._shared_mem = multiprocessing.Array(ctypes.c_float, count)
        self.noise = np.ctypeslib.as_array(self._shared_mem.get_obj())
        assert self.noise.dtype == np.float32
        self.noise[:] = np.random.RandomState(seed).randn(count)  # 64-bit to 32-bit conversion here
        print(f'Sampled {self.noise.size * 4} bytes')

    def get(self, i, dim):
        return self.noise[i:i + dim]

    def sample_index(self, stream, dim):
        return stream.randint(0, len(self.noise) - dim + 1)


def compute_ranks(x):
    """
    Returns ranks in [0, len(x))
    Note: This is different from scipy.stats.rankdata, which returns ranks in [1, len(x)].

    Returns array of integers indicating the rank of the original value
    """
    assert x.ndim == 1
    ranks = np.empty(len(x), dtype=int)
    ranks[x.argsort()] = np.arange(len(x))
    return ranks


def compute_centered_ranks(x):
    """
    args:
        -x: [pop_size, 2] array of total rewards for pop_size of mutations (+deltas and -deltas)
    returns:
        -y: fitness values of the same shape, but ranked and centered around 0 (interval <-0.5, 0.5>
    """
    # ravel: contiguous flattened array
    y = compute_ranks(x.ravel()).reshape(x.shape).astype(np.float32)
    y /= (x.size - 1)   # to <0,1>
    y -= .5             # to <-0.5, 0.5>
    return y


def itergroups(items, group_size):
    """ iterator for batches
    """
    assert group_size >= 1
    group = []
    for x in items:
        group.append(x)
        if len(group) == group_size:
            yield tuple(group)
            del group[:]
    if group:
        yield tuple(group)


def batched_weighted_sum(weights, vecs, batch_size):
    """
    args:
        -weights: relative quality of direction (+delta vs -delta)
        -vecs: deltas (read from the noise table)
        -batch_size: int
    returns:
        -result, number of items summed
    """
    total = 0.
    num_items_summed = 0

    # build weights and genomes of batch_size
    for batch_weights, batch_vecs in zip(itergroups(weights, batch_size), itergroups(vecs, batch_size)):
        assert len(batch_weights) == len(batch_vecs) <= batch_size

        total += np.dot(
            np.asarray(batch_weights, dtype=np.float32),  # [batch_size, genome_length] - genomes
            np.asarray(batch_vecs, dtype=np.float32)      # [batch_size]                - relative fitnesses
        )
        num_items_summed += len(batch_weights)
    return total, num_items_summed


