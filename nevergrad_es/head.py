import multiprocessing
import time
from argparse import Namespace
import numpy as np
from badger_utils.sacred import SacredWriter
from tqdm import tqdm

from utils.eval import build_policy_env, eval_genome
from es.head import try_deserialize
import nevergrad as ng
from nevergrad_es.worker import RayNevergradWorker, update_best
import ray


def run_nevergrad(conf: Namespace, _run=None, writer: SacredWriter = None):
    """ Optimization using the Nevergrad library and workers parallelized using Ray.
    """
    print(f'HEAD: started...')
    policy, _ = build_policy_env(conf)
    genome_length = policy.serialize_to_genome().size
    try_deserialize(conf, policy)

    budget = conf.pop_size * conf.num_generations
    parallelization = multiprocessing.cpu_count()
    serialization_period = max(conf.num_generations // conf.num_serializations, 1)
    assert conf.seed is not None, 'for Nevergrad, only integer seeds are supported (-1 or positive int)'

    print(f'HEAD: Nevergrad will use tool: {conf.tool}, genome length: {genome_length}, budget: {budget}')
    optimizer = ng.optimizers.registry[conf.tool](parametrization=genome_length, budget=budget, num_workers=1)

    workers = [RayNevergradWorker.remote() for _ in range(parallelization)]

    overall_best_f = None
    total_sim_steps = 0
    gen = 0
    fitness_values = []  # all values in this "generation"
    num_env_steps = []

    exp_start = time.time()
    gen_start = exp_start

    # https://facebookresearch.github.io/nevergrad/machinelearning.html?highlight=machine%20learn#optimization-of-parameters-for-reinforcement-learning
    for _ in tqdm(range(budget // parallelization)):

        """Ask for genomes"""
        individuals = [optimizer.ask() for _ in range(parallelization)]

        """Evaluate"""
        if conf.use_ray:
            for ind, worker in zip(individuals, workers):  # send the task to the workers
                worker.submit_genome.remote(conf, ind.args[0], conf.seed)

            future_results = [worker.collect_fitness.remote() for worker in workers]  # collect result pointers
            results = ray.get(future_results)  # wait for results
        else:
            results = []
            for ind in individuals:
                results.append(eval_genome(conf, ind.args[0], conf.seed))

        fitness_results = [result[0] for result in results]

        """Tell the fitness values"""
        for ind, fitness in zip(individuals, fitness_results):
            optimizer.tell(ind, -fitness)  # Nevergad minimizes

        """Collect stats"""
        fitness_values.extend(fitness_results)
        num_env_steps.extend([result[1] for result in results])

        """Publish stats"""
        if len(fitness_values) >= conf.pop_size:

            current_gen_fitness_values = fitness_values[:conf.pop_size]
            current_gen_num_steps = num_env_steps[:conf.pop_size]

            fitness_values = fitness_values[conf.pop_size:]  # remove the processed results
            num_env_steps = num_env_steps[conf.pop_size:]

            mean_fitness = sum(current_gen_fitness_values) / conf.pop_size
            mean_num_steps = sum(current_gen_num_steps) / conf.pop_size
            total_sim_steps += sum(current_gen_num_steps)

            best_fitness = max(current_gen_fitness_values)
            overall_best_f = update_best(overall_best_f, current_gen_fitness_values)

            # obtain recommended genome and eval its fitness
            current_genome = optimizer.recommend().args[0]
            policy.deserialize_from_genome(current_genome)
            current_fitness, _ = eval_genome(conf, current_genome, conf.env_seed)

            now = time.time()

            _run.log_scalar('gen', gen)
            _run.log_scalar('total time elapsed', now - exp_start)
            _run.log_scalar('time per gen', now - gen_start)

            _run.log_scalar('total sim steps', total_sim_steps)
            _run.log_scalar('total fitness evals', conf.pop_size * gen)
            _run.log_scalar('mean episode length', mean_num_steps / conf.num_episodes)
            _run.log_scalar('mean steps per eval', mean_num_steps)

            _run.log_scalar('current fitness', current_fitness)
            _run.log_scalar('mean fitness', mean_fitness)
            _run.log_scalar('best fitness', best_fitness)
            _run.log_scalar('overall best fitness', overall_best_f)
            _run.log_scalar('fitness st.dev', float(np.array(current_gen_fitness_values).std()))

            _run.log_scalar('norm', float(np.square(current_genome).sum()))
            _run.log_scalar('genome length', genome_length)

            if writer is not None and gen % serialization_period == 0:
                policy.save(writer=writer, epoch=gen)

            gen += 1
            gen_start = now

