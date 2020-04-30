import random
import time
from argparse import Namespace
from typing import Optional, Tuple

import gym
import pybulletgym
import numpy as np

from env.partially_observable_cartpole import PartiallyObservableCartPole, PartiallyObservableCartPoleEasy
from env.partially_observable_pendulum import PartiallyObservablePendulum
from env.sphere import Sphere
from policy.policy import Policy
from utils.available_device import choose_device
from env.gym_utils import fix_observation, sanitize_reward


def simulate(env: gym.Env,
             policy: Policy,
             num_episodes: int,  # ~ num episodes to be averaged over
             render: bool = False,
             sleep_render: float = 0.0,
             max_ep_length: Optional[int] = None) -> Tuple[float, int]:
    """Used both for evaluation and rendering"""

    if render:  # pybullet compatibility (render before reset), other environments need reset before render..
        try:
            env.render()
        except:
            pass

    fitness_values = []  # reward over all episodes
    episode_rewards = []  # reward over one episode
    num_steps_used = 0
    episode = -1
    done = True
    observation = None
    step = 0

    policy.reset()

    while True:
        step += 1

        # artificially kill the episode?
        if max_ep_length is not None and step > max_ep_length:
            done = True

        # reset between episodes
        if done:
            step = 0
            episode += 1
            # compute the mean episode fitness
            if len(episode_rewards) > 0:
                # print(f'done and last reward is {episode_rewards[-1]} and num steps is {len(episode_rewards)}')
                fitness_values.append(sum(episode_rewards))
                # time.sleep(2)
            num_steps_used += len(episode_rewards)
            episode_rewards = []

            observation = fix_observation(env.reset())
            policy.reset()
            if render:
                print(f'\n ---------------- reset after episode {episode}!\n')

            # all episodes done?
            if episode >= num_episodes:
                fitness = sum(fitness_values) / num_episodes
                return fitness, num_steps_used

        # The main simulation loop
        action = policy.pick_action(observation)
        observation, reward, done, _ = env.step(action)

        observation = fix_observation(observation)
        episode_rewards.append(sanitize_reward(reward))

        if render:
            r = sanitize_reward(reward if reward is not None else 0)
            print(f'step: {step}\tobs: {observation.reshape(-1)},\ta: {action},\tr: {r},\t done: {done}')
            env.render()
            if sleep_render > 0.0:
                time.sleep(sleep_render)


def create_env_seed(env_seed: int) -> int:
    """ Initialize seed for the environment, -1 means new random seed"""
    if env_seed != -1:
        return env_seed

    random.seed(time.time())
    return random.randint(0, 2 ** 31)


def build_policy_env(conf: Namespace, seed: int = -1) -> Tuple[Policy, gym.Env]:
    """Build policy and environment, seed=-1 means random seed"""
    choose_device(conf)

    if conf.env_name == 'sphere':
        env = Sphere()
    elif conf.env_name == 'PartiallyObservablePendulum':
        env = PartiallyObservablePendulum(gym.make('Pendulum-v0'))
    elif conf.env_name == 'PartiallyObservableCartPole':
        env = PartiallyObservableCartPole(gym.make('CartPole-v0'))
    elif conf.env_name == 'PartiallyObservableCartPoleEasy':
        env = PartiallyObservableCartPoleEasy(gym.make('CartPole-v0'))
    else:
        env = gym.make(conf.env_name)

    seed = create_env_seed(seed)
    env.seed(seed)

    policy = Policy(env.observation_space, env.action_space, conf)
    return policy, env


def eval_genome(conf: Namespace,
                genome: np.ndarray,
                seed: int) -> Tuple[float, float]:
    """ Evaluates genome using the experiment configuration

    Args:
        conf: Namespace - experiment config
        genome: np.ndarray of floats defining all the optimized parameters
        seed: to be used in the initialization of the environment (-1 means random)
    """
    policy, env = build_policy_env(conf, seed)
    policy.deserialize_from_genome(genome)

    fitness, num_steps_used = simulate(env=env,
                                       policy=policy,
                                       num_episodes=conf.num_episodes,
                                       render=conf.render,
                                       sleep_render=conf.sleep,
                                       max_ep_length=conf.max_ep_length)

    env.close()  # TODO this does not work in some environments (memory leak), reuse of instances would be better
    return fitness, num_steps_used
