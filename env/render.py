import time
from argparse import Namespace
from pathlib import Path
from typing import Optional

import click
from badger_utils.sacred import SacredReader

from utils.eval import simulate, build_policy_env
from utils.sacred_local import get_sacred_storage


@click.command()
@click.argument('exp_id', type=int)
@click.option('--gen', default=None, help='Generation to be deserialized (the last one by default)')
@click.option('--sleep', default=0.001, help='Sleep time between the time steps')
@click.option('--num_episodes', default=None, type=int, help='Override num_episodes parameter?')
@click.option('--max_ep_length', default=None, type=int, help='Override the max_episode_length?')
def render(exp_id: int,
           gen: int,
           sleep: float,
           num_episodes: Optional[int],
           max_ep_length: Optional[int]):
    """Download a given config and policy from the sacred, run the inference"""

    # parse arguments, init the reader
    reader = SacredReader(exp_id, get_sacred_storage(), data_dir=Path.cwd())

    # obtain the config
    config = Namespace(**reader.config)
    num_episodes = num_episodes if num_episodes is not None else config.num_episodes
    max_ep_length = max_ep_length if max_ep_length is not None else config.max_ep_length
    env_seed = config.env_seed if config.env_seed is not None else -1

    policy, env = build_policy_env(config, env_seed)

    # deserialize the model parameters
    if gen is None:
        gen = reader.find_last_epoch()

    print(f'Deserialization from the epoch: {gen}')
    time.sleep(2)
    policy.load(reader=reader, epoch=gen)

    fitness, num_steps_used = simulate(env=env,
                                       policy=policy,
                                       num_episodes=num_episodes,
                                       max_ep_length=max_ep_length,
                                       render=True,
                                       sleep_render=sleep)

    print(f'\n\n Done, fitness is: {fitness}, num_steps: {num_steps_used}\n\n')


if __name__ == '__main__':
    render()
