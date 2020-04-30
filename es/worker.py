import os
import time
from typing import Dict, Optional

import click
import numpy as np

from cluster.redis_clients import WorkerClient, RelayClient
from es.es_utils.es_common import SharedNoiseTable, Result, TaskData, LOCAL_RELAY_SOCKET_PATH
from utils.eval import build_policy_env, eval_genome, create_env_seed


def setup_experiment(redis_client: WorkerClient):
    start_time, conf = redis_client.get_experiment()
    policy, env = build_policy_env(conf)
    genome_length = policy.serialize_to_genome().size

    rs = np.random.RandomState()
    worker_id = rs.randint(2 ** 31)

    print(f'Worker {worker_id} here, optimizing theta of size: {genome_length}')

    return start_time, conf, policy, env, genome_length


def run_worker(redis_config: Dict, noise: Optional[SharedNoiseTable] = None):
    """Evaluate mutated genome on the sphere function for num_steps and return result (genome, fitness)"""

    if noise is None:
        noise = SharedNoiseTable()
    rs = np.random.RandomState()
    worker_id = rs.randint(2 ** 31)

    redis_client = WorkerClient(redis_config)
    start_time, config, policy, my_envs, genome_size = setup_experiment(redis_client)

    while True:

        get_task_start = time.time()
        task_id, task_data = redis_client.get_current_task()
        if task_id.start_time != start_time:
            print(f'WORKER: received task from invalid experiment ({start_time} != {task_id.start_time})')
            start_time, config, policy, my_envs, genome_size = setup_experiment(redis_client)
            print(f'WORKER: obtained new experiment with start_time: {start_time}')
            task_id, task_data = redis_client.get_current_task()
            start_time = task_id.start_time

        time_to_obtain_task = time.time() - get_task_start

        print(f'WORKER: obtained task: {task_id}')
        assert isinstance(task_data, TaskData)
        assert task_data.genome.size == genome_size
        assert isinstance(task_data.env_seed, int)

        task_tstart = time.time()
        noise_inds, returns, lengths = [], [], []

        """Collect evaluations for some reasonable min_task_runtime (so that we are not communicating so often)"""
        while not noise_inds or time.time() - task_tstart < config.min_task_runtime:
            # print(f'WORKER {worker_id} eval started')

            # get a pseudo-random index in the noise table, read the noise vector and rescale using the stdev..
            noise_idx = noise.sample_index(rs, policy.num_params)
            deltas = config.sigma * noise.get(noise_idx, policy.num_params)
            seed = create_env_seed(task_data.env_seed)  # seed has to be the same for mirroring

            # mutate in both direction, evaluate each
            genome = task_data.genome + deltas
            rew_pos, len_pos = eval_genome(conf=config, genome=genome, seed=seed)

            genome = task_data.genome - deltas
            rew_neg, len_neg = eval_genome(conf=config, genome=genome, seed=seed)

            # print(f'rew_pos: {rew_pos}, len: {len_pos} rew_neg: {rew_neg}, len {len_neg}')
            # print(f'WORKER {worker_id} eval ended')

            # collect the results
            noise_inds.append(noise_idx)
            returns.append([rew_pos, rew_neg])
            lengths.append([len_pos, len_neg])

        time_to_eval_task = time.time() - task_tstart

        # compose the result
        result = Result(
            worker_id=worker_id,
            noise_inds_n=np.array(noise_inds),  # [N] indexes in the noise table
            returns_n2=np.array(returns, dtype=np.float32),  # [N, 2] fitnesses (+delta, -delta)
            lengths_n2=np.array(lengths, dtype=np.int32),  # [N, 2] env. steps for each evaluation
            time_to_eval=time_to_eval_task,
            time_to_obtain=time_to_obtain_task
        )

        # print(f'WORKER: pushing {len(noise_inds)} result(s).')
        # push to the server
        redis_client.push_result(task_id=task_id, result=result)


@click.group()
def cli():
    pass


@cli.command()
@click.option('--force_relay/--no_force_relay', help='force redis relay even if running locally?')
@click.option('--num_workers', type=int, default=1)
def start_workers(force_relay: bool, num_workers: int):
    """Runs just n workers locally"""

    if force_relay:
        print(f'\n\nRunning local version WITH relay\n\n')
        redis_config: Dict = {'unix_socket_path': '/tmp/es_redis_relay.sock'}
    else:
        print(f'\n\nRunning local version WITHOUT relay\n\n')
        redis_config: Dict = {'unix_socket_path': '/tmp/es_redis_master.sock'}

    time.sleep(3)
    if num_workers == 1:
        run_worker(redis_config)
    else:
        noise = SharedNoiseTable()
        for _ in range(num_workers):
            if os.fork() == 0:
                run_worker(redis_config, noise)
                return
        os.wait()


@cli.command()
@click.option('--master_host', default='localhost', type=str)
@click.option('--master_port', default=6379, type=int)
@click.option('--relay_socket_path', default=LOCAL_RELAY_SOCKET_PATH, type=str)
def run_relay(master_host: str, master_port: int, relay_socket_path: str):
    """Runs a relay locally"""
    master_port = int(master_port)

    # Start the relay
    master_redis_cfg = {'host': master_host, 'port': master_port}
    relay_redis_cfg = {'unix_socket_path': relay_socket_path}

    # Fork a child process. Return 0 to child process and PID of child to parent process.
    RelayClient(master_redis_cfg, relay_redis_cfg).run()
    os.wait()


@cli.command()
def run_mirror():
    """Runs a local redis mirror"""
    print(f'\n\nStarting redis mirror!\n\n')
    os.system('redis-server cluster/redis_config/redis_local_mirror.conf')


@cli.command()
@click.option('--master_host', default='localhost', type=str)
@click.option('--master_port', default=6379, type=int)
@click.option('--relay_socket_path', default=LOCAL_RELAY_SOCKET_PATH, type=str)
@click.option('--num_workers', type=int, default=1)
def workers(master_host: str, master_port: int, relay_socket_path: str, num_workers: int):
    """Runs num_workers in a worker instance (cluster usage)"""

    print(f'Hello, workers here')

    # Start the relay
    master_redis_cfg = {'host': master_host, 'port': master_port}
    relay_redis_cfg = {'unix_socket_path': relay_socket_path}

    if os.fork() == 0:
        RelayClient(master_redis_cfg, relay_redis_cfg).run()
        return

    if num_workers == 1:
        run_worker(relay_redis_cfg)
    else:
        noise = SharedNoiseTable()
        for _ in range(num_workers):
            if os.fork() == 0:
                run_worker(relay_redis_cfg, noise)
                return
        os.wait()


if __name__ == '__main__':
    cli()
