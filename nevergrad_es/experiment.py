from argparse import Namespace
import ray
from badger_utils.sacred import SacredWriter, SacredConfigFactory
from sacred import SETTINGS, Experiment

from default_params import default_common_params, default_nevergrad_params
from nevergrad_es.head import run_nevergrad
from utils.sacred_local import get_sacred_storage

ex = Experiment("Nevergrad")
sacred_writer = SacredWriter(ex, get_sacred_storage())
SETTINGS.CONFIG.READ_ONLY_CONFIG = False  # config sent over network
SETTINGS['CAPTURE_MODE'] = 'sys'  # some error after experiment with Ray finishes


@ex.config
def default_config():
    ex.add_config(default_common_params)
    ex.add_config(default_nevergrad_params)


@ex.named_config
def lander():
    """Solvable in tens of minutes on the PC, just two neurons."""
    hidden_sizes = []
    output_activation = 'none'
    pop_size = 20
    num_generations = 300

    num_episodes = 5  # this is important here


@ex.named_config
def ant():
    """Network from the paper, hyperparams different, usually works well"""
    env_name = 'AntPyBulletEnv-v0'
    hidden_sizes = [64, 64]
    hidden_activation = 'tanh'
    output_activation = 'discrete'

    num_generations = 1000
    pop_size = 100

    max_ep_length = 500  # log enough to be beneficial to run, but not too long

    num_episodes = 5  # this is important here


@ex.named_config
def pendulum():
    """Pendulum task, works well with this config (although tanh on the output is a problem)."""
    env_name = 'Pendulum-v0'
    hidden_sizes = [64, 64]
    hidden_activation = 'relu'
    output_activation = 'none'

    max_ep_length = 1000

    num_episodes = 5
    env_seed = -1


@ex.named_config
def pendulum_lstm():
    """LSTM on fully observable pendulum, works (converges in around 250 gens with this config: 4616)."""
    env_name = 'Pendulum-v0'
    network = 'policy.lstm_network.LSTMNetwork'
    hidden_sizes = [5]
    output_activation = 'none'

    env_seed = -1
    num_episodes = 5


@ex.named_config
def pendulum_partial():
    """LSTM on partially observable task, only CMA seems to work now."""
    env_name = 'PartiallyObservablePendulum'
    hidden_sizes = [16]
    output_activation = 'none'
    network = 'policy.lstm_network.LSTMNetwork'
    tool = 'CMA'

    env_seed = -1
    num_episodes = 5

    num_generations = 700
    pop_size = 50


@ex.automain
def run(_run, _config):
    config = Namespace(**_config)
    ray.init()
    run_nevergrad(config, _run, sacred_writer)

