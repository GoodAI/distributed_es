import os
import time
from argparse import Namespace
from pathlib import Path
from typing import Dict, Tuple

import click
import numpy as np

from cluster.redis_clients import MasterClient
from es.es_utils.es_common import SharedNoiseTable, compute_centered_ranks, batched_weighted_sum, Result, TaskData
from es.es_utils.optimizers import Adam
from policy.policy import Policy
from utils.eval import build_policy_env, eval_genome
from utils.sacred_local import get_sacred_storage

EXP_KEY = 'es:exp'
TASK_ID_KEY = 'es:task_id'
TASK_DATA_KEY = 'es:task_data'
TASK_CHANNEL = 'es:task_channel'
RESULTS_KEY = 'es:results'


def try_deserialize(config: Namespace, policy: Policy):
    """Deserialize the policy params if configured to do so"""
    if config.load_from is not None:
        print(f'DESERIALIZATION: will load policy from the exp-id {config.load_from}')
        from badger_utils.sacred import SacredReader
        reader = SacredReader(config.load_from, get_sacred_storage(), data_dir=Path.cwd())
        policy.load(reader=reader, epoch=reader.find_last_epoch())


def _pop_result(redis_master: MasterClient,
                generation: int,
                start_time: float) -> Tuple[Result, int]:
    skipped = 0

    while True:
        task_id, result = redis_master.pop_result()
        if task_id.start_time != start_time:
            print(f'HEAD: received task_id is not from this experiment ({start_time}), skipping ({task_id.start_time})')
            continue
        elif task_id.generation != generation:
            skipped += 1
            continue

        return result, skipped


def compute_success_rate(current_f: float, returns_n2: np.ndarray) -> float:
    num_better = 0
    for f in returns_n2:
        if f[0] > current_f or f[1] > current_f:
            num_better += 1
    return num_better / returns_n2.size


def run_head(redis_config: Dict, conf: Namespace, _run=None, writer=None):
    """ Parse config, instantiate the policy etc.. """
    print(f'HEAD: started..')
    assert isinstance(conf, Namespace)
    if conf.env_seed is not None and conf.env_seed != -1 and conf.num_episodes > 1:
        print(f'WARNING: num_episoses can be 1 in case the env_seed is fixed to constant value')
    policy, _ = build_policy_env(conf)
    # best_policy, _ = build_policy_env(conf)
    genome_length = policy.serialize_to_genome().size
    try_deserialize(conf, policy)
    print(f'HEAD: genome length: {genome_length}')

    """ Initialize ES-related components """
    # optimizer = SGD(pi=policy, stepsize=conf.lr, momentum=0.9)
    optimizer = Adam(pi=policy, stepsize=conf.lr, beta1=conf.adam_beta1, beta2=conf.adam_beta2,
                     epsilon=conf.adam_epsilon)
    serialization_period = max(conf.num_generations // conf.num_serializations, 1)

    noise = SharedNoiseTable()
    rs = np.random.RandomState()

    tstart = time.time()

    total_sim_steps = 0
    total_fitness_evals = 0
    overall_best_f = None
    current_f = None

    """ Initialize redis and declare experiment """
    redis_master = MasterClient(redis_config)
    redis_master.flushall()
    start_time = redis_master.declare_experiment(conf)

    for gen in range(conf.num_generations):

        print(f'HEAD: submitting task {gen}')
        current_genome = policy.serialize_to_genome()

        # All individuals from this generation will have the same seed
        task_data = TaskData(genome=current_genome,
                             env_seed=rs.randint(2 ** 31) if conf.env_seed is None else conf.env_seed)
        redis_master.declare_task(task_id=gen, task_data=task_data)
        results = []
        num_results = 0
        worker_ids = []
        current_gen_results = []
        skipped_results = 0

        time_to_obtain = 0.0
        time_to_eval = 0.0

        gen_tstart = time.time()

        if _run is not None:
            current_f, _ = eval_genome(conf=conf, genome=current_genome,
                                       seed=rs.randint(2 ** 31) if conf.env_seed is None else conf.env_seed)

        """Collect the pop_size evaluated mutated individuals"""
        while num_results < conf.pop_size:
            result, skipped = _pop_result(redis_master, gen, start_time)
            results.append(result)

            # check the data
            assert isinstance(result, Result)
            assert (result.noise_inds_n.ndim == 1 and
                    result.returns_n2.shape == result.lengths_n2.shape == (len(result.noise_inds_n), 2))
            assert result.returns_n2.dtype == np.float32

            # update counts
            worker_ids.append(result.worker_id)
            num_results += result.returns_n2.size
            skipped_results += skipped
            time_to_eval += result.time_to_eval
            time_to_obtain += result.time_to_obtain
            # best_f = _update_best_policy(best_policy, policy, best_f, result, noise, conf.sigma)

            current_gen_results.append(result)
            # print(f'HEAD: popped result [num results: {num_results}/{conf.pop_size}] (popped {len(results)} times)')

        """Assemble the results"""
        noise_inds_n = np.concatenate([r.noise_inds_n for r in current_gen_results])
        returns_n2 = np.concatenate([r.returns_n2 for r in current_gen_results])
        lengths_n2 = np.concatenate([r.lengths_n2 for r in current_gen_results])
        assert noise_inds_n.shape[0] == returns_n2.shape[0] == lengths_n2.shape[0]

        print(f'HEAD: popped {num_results} new results ({skipped_results} skipped),'
              f' generation: {gen} (total num. evals: {returns_n2.size})')

        """Fitness shaping, obtaining mutations, gradient computation"""
        proc_returns_n2 = compute_centered_ranks(returns_n2)
        mutations = [noise.get(idx, policy.num_params) for idx in noise_inds_n]

        # Compute gradient by reading mutation vectors from the shared table, and take step
        grad, count = batched_weighted_sum(
            weights=proc_returns_n2[:, 0] - proc_returns_n2[:, 1],  # relative quality of the delta directions
            vecs=mutations,
            batch_size=500)

        # normalize
        grad /= returns_n2.size
        assert grad.shape == (policy.num_params,) and grad.dtype == np.float32 and count == len(noise_inds_n)

        # apply the gradient
        update_ratio = optimizer.update(-grad + conf.l2coeff * current_genome)

        """ Publish stats"""
        if _run is not None:

            best_f = np.max(returns_n2)
            if overall_best_f is None or best_f > overall_best_f:
                overall_best_f = best_f

            success_rate = compute_success_rate(current_f, returns_n2)
            total_sim_steps += lengths_n2.sum()
            total_fitness_evals += lengths_n2.size

            step_tend = time.time()

            _run.log_scalar('gen', gen)
            _run.log_scalar('total time elapsed', step_tend - tstart)
            _run.log_scalar('time per gen', step_tend - gen_tstart)

            _run.log_scalar('total sim steps', total_sim_steps)
            _run.log_scalar('total fitness evals', total_fitness_evals)
            _run.log_scalar('mean episode length', lengths_n2.mean() / conf.num_episodes)
            _run.log_scalar('mean steps per eval', lengths_n2.mean())

            _run.log_scalar('current fitness', current_f)  # individual that will be used now
            _run.log_scalar('mean fitness', returns_n2.mean())  # mean fitness of the current generation
            _run.log_scalar('best fitness', best_f)  # best in the current gen
            _run.log_scalar('overall best fitness', overall_best_f)
            _run.log_scalar('fitness st.dev', float(returns_n2.std()))

            _run.log_scalar('norm', float(np.square(policy.serialize_to_genome()).sum()))
            _run.log_scalar('genome length', genome_length)  # constant

            # other stats
            _run.log_scalar("success rate", success_rate)
            _run.log_scalar('grad norm', float(np.square(grad).sum()))
            _run.log_scalar('update ratio', float(update_ratio))

            _run.log_scalar("frac. results skipped",
                            conf.pop_size / skipped_results if skipped_results != 0 else 0)
            _run.log_scalar("results skipped", skipped_results)
            num_unique_workers = len(set(worker_ids))
            _run.log_scalar('num unique workers', num_unique_workers)
            _run.log_scalar("worker used times", len(worker_ids) / num_unique_workers)

            _run.log_scalar('time_to_eval', time_to_eval / len(results))  # how long it takes to evaluate 1 Result
            _run.log_scalar('time_to_obtain', time_to_obtain / len(results))  # how long it takes to pop the Task

            if writer is not None:
                if serialization_period is not None and gen % serialization_period == 0:
                    # best_policy.save(writer=writer, epoch=2 * gen)
                    # policy.save(writer=writer, epoch=2 * gen + 1)
                    policy.save(writer=writer, epoch=gen)

    redis_master.flushall()  # delete the task from the database, so that workers stop computing
    return policy.serialize_to_genome()


@click.group()
def cli():
    pass


@cli.command()
@click.option('--cluster/--local', help='Running locally or in a cluster?')
def flush(cluster: bool):
    if cluster:
        print(f'Flushing cluster DB')
        redis_config: Dict = {'unix_socket_path': '/var/run/redis/redis.sock'}
    else:
        print(f'Flushing local DB')
        redis_config: Dict = {'unix_socket_path': '/tmp/es_redis_master.sock'}

    redis_master = MasterClient(redis_config)
    redis_master.flushall()
    print(f'Redis DB flushed, workers should stop now')


@cli.command()
@click.option('--stop/--not-stop', help='Stop a currently running redis-server?')
def run_server(stop: bool):
    """Runs the redis server"""
    if stop:
        print(f'\n\nStopping the redis-server!')
        os.system('service redis-server stop')
    print(f'\n\nStarting redis server!\n\n')
    os.system('redis-server cluster/redis_config/redis_master.conf')


if __name__ == '__main__':
    cli()
