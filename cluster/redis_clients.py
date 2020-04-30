import os
import pickle
import time
from collections import deque
from typing import Tuple, List, Union

import redis

from es.es_utils.es_common import TaskId

EXP_KEY = 'es:exp'
TASK_ID_KEY = 'es:task_id'
TASK_DATA_KEY = 'es:task_data'
TASK_CHANNEL = 'es:task_channel'
RESULTS_KEY = 'es:results'


def serialize(x):
    return pickle.dumps(x, protocol=-1)


def deserialize(x):
    return pickle.loads(x)


def retry_connect(redis_cfg, tries=300, base_delay=4.):
    """Connects to the redis server"""
    for i in range(tries):
        try:
            r = redis.StrictRedis(**redis_cfg)
            r.ping()
            return r
        except redis.ConnectionError as e:
            if i == tries - 1:
                raise
            else:
                delay = base_delay * (1 + (os.getpid() % 10) / 9)
                print(f'WARNING: could not connect to {redis_cfg}. Retrying after {delay} sec ({i+2}/{tries}). Error {e}')
                time.sleep(delay)


class MasterClient:
    start_time: float

    """Used by the head to declare_task and pop_results"""
    def __init__(self, master_redis_cfg):
        self.master_redis = retry_connect(master_redis_cfg)
        self.start_time = -1.
        print(f'[master] Connected to Redis: {self.master_redis}')

    def declare_experiment(self, exp_config) -> float:
        self.start_time = time.time()
        self.master_redis.set(EXP_KEY, serialize((self.start_time, exp_config)))
        # print(f'[master] Declared experiment {pformat(exp_config)}')
        return self.start_time

    def declare_task(self, task_id: int, task_data):
        assert self.start_time != -1., 'ERROR: experiment not declared!'
        serialized_task_data = serialize(task_data)
        task_id = TaskId(start_time=self.start_time, generation=task_id)
        serialized_task_id = serialize(task_id)

        (self.master_redis.pipeline()
         .mset({TASK_ID_KEY: serialized_task_id,
                TASK_DATA_KEY: serialized_task_data})
         .publish(TASK_CHANNEL, serialize((serialized_task_id, serialized_task_data)))
         .execute())

        # print(f'[master] Declared task gen: {task_id.generation} \t({task_id.start_time})')

    def pop_result(self) -> Tuple[TaskId, any]:
        task_id, result = deserialize(self.master_redis.blpop(RESULTS_KEY)[1])

        # print(f'[master] Popped a result for gen {task_id.generation}\t ({task_id.start_time})')
        return task_id, result

    def flushall(self):
        self.master_redis.flushall(asynchronous=False)


def retry_get(pipe, key: Union[List, Tuple, str], tries: int = 3000, base_delay: float = 4.):
    for i in range(tries):
        # Try to (m)get
        if isinstance(key, (list, tuple)):
            vals = pipe.mget(key)
            if all(v is not None for v in vals):
                return vals
        else:
            val = pipe.get(key)
            if val is not None:
                return val
        # Sleep and retry if any key wasn't available
        if i != tries - 1:
            delay = base_delay * (1 + (os.getpid() % 10) / 9)
            print(f'Key: {key} not set. Retrying after {delay} sec ({i+2}/{tries})')
            time.sleep(delay)
    raise RuntimeError('{} not set'.format(key))


class RelayClient:
    """
    Receives and stores task broadcasts from the master
    Batches and pushes results from workers to the master
    """

    def __init__(self, master_redis_cfg, relay_redis_cfg):
        self.master_redis = retry_connect(master_redis_cfg)
        print(f'[relay] Connected to master: {self.master_redis}')
        self.local_redis = retry_connect(relay_redis_cfg)
        print(f'[relay] Connected to relay: {self.local_redis}')

    def run(self):
        # Initialization: read exp and latest task from master
        self.local_redis.set(EXP_KEY, retry_get(self.master_redis, EXP_KEY))
        self._declare_task_local(*retry_get(self.master_redis, (TASK_ID_KEY, TASK_DATA_KEY)))

        # Start subscribing to tasks
        p = self.master_redis.pubsub(ignore_subscribe_messages=True)
        p.subscribe(**{TASK_CHANNEL: lambda msg: self._declare_task_local(*deserialize(msg['data']))})
        p.run_in_thread(sleep_time=0.001)

        # Loop on RESULTS_KEY and push to master
        batch_sizes, last_print_time = deque(maxlen=20), time.time()  # for logging
        while True:
            results = []
            start_time = curr_time = time.time()
            while curr_time - start_time < 0.001:
                results.append(self.local_redis.blpop(RESULTS_KEY)[1])
                curr_time = time.time()
            self.master_redis.rpush(RESULTS_KEY, *results)
            # Log
            batch_sizes.append(len(results))
            if curr_time - last_print_time > 5.0:
                print(f'[relay] Average batch size {sum(batch_sizes)/ len(batch_sizes)}')
                last_print_time = curr_time

    def _declare_task_local(self, task_id, task_data):
        print(f'[relay] Received task {task_id}')
        self.local_redis.mset({TASK_ID_KEY: task_id, TASK_DATA_KEY: task_data})


class WorkerClient:
    def __init__(self, relay_redis_cfg):
        self.local_redis = retry_connect(relay_redis_cfg)
        print(f'[worker] Connected to redis relay: {self.local_redis}')

        self.cached_task_id, self.cached_task_data = None, None

    def get_experiment(self) -> Tuple[float, any]:
        start_time, exp_config = deserialize(retry_get(self.local_redis, EXP_KEY))
        print(f'[worker] Deserialized experiment: {exp_config}, start time {start_time}')
        return start_time, exp_config

    def get_current_task(self) -> Tuple[TaskId, any]:
        """Get task to be computed"""
        with self.local_redis.pipeline() as pipe:
            while True:
                try:
                    # optimistic locking: https://realpython.com/python-redis/#using-redis-py-redis-in-python
                    pipe.watch(TASK_ID_KEY)
                    task_id: TaskId = deserialize(retry_get(pipe, TASK_ID_KEY))
                    if task_id == self.cached_task_id:
                        # print(f'[worker] Returning cached task {task_id}')
                        break
                    pipe.multi()
                    pipe.get(TASK_DATA_KEY)
                    # print(f'[worker] Getting new task {task_id}. Cached task was {self.cached_task_id}')
                    self.cached_task_id, self.cached_task_data = task_id, deserialize(pipe.execute()[0])
                    break
                except redis.WatchError:
                    continue
        return self.cached_task_id, self.cached_task_data

    def push_result(self, task_id: TaskId, result):
        self.local_redis.rpush(RESULTS_KEY, serialize((task_id, result)))
        # print(f'[worker] Pushed result for task {task_id}')

