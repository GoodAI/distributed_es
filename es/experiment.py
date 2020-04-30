from argparse import Namespace
# from badger_utils import SacredWriter, SacredConfigFactory
from badger_utils.sacred import SacredWriter, SacredConfigFactory
from sacred import SETTINGS, Experiment
from default_params import default_es_params, default_common_params
from es.es_utils.es_common import LOCAL_MASTER_SOCKET_PATH, REMOTE_MASTER_SOCKET_PATH
from es.head import run_head
from utils.sacred_local import get_sacred_storage

ex = Experiment("ES")
sacred_writer = SacredWriter(ex, get_sacred_storage())
SETTINGS.CONFIG.READ_ONLY_CONFIG = False  # config sent over network


@ex.config
def default_config():
    ex.add_config(default_common_params)
    ex.add_config(default_es_params)


@ex.named_config
def lander():
    """Solvable in tens of minutes on the PC, just two neurons."""
    hidden_sizes = []
    output_activation = 'none'
    pop_size = 20
    num_generations = 300

    num_episodes = 1

    lr = 0.1
    sigma = 0.1


@ex.named_config
def ant():
    """Network from the paper, hyperparams are a bit different, usually works well"""
    env_name = 'AntPyBulletEnv-v0'
    hidden_sizes = [64, 64]
    hidden_activation = 'tanh'
    output_activation = 'discrete'

    pop_size = 100
    num_episodes = 1
    lr = 0.01
    sigma = 0.1

    num_generations = 1000

    max_ep_length = 500  # log enough to be beneficial to run, but not too long


@ex.named_config
def pendulum():
    """Pendulum task, works well with this config."""
    env_name = 'Pendulum-v0'
    hidden_sizes = [64, 64]
    hidden_activation = 'relu'
    output_activation = 'none'

    pop_size = 200
    lr = 0.05
    sigma = 0.07

    num_generations = 700

    num_episodes = 5
    env_seed = -1


@ex.named_config
def pendulum_lstm():
    """LSTM on fully observable pendulum, works (converges in around 250 gens with this config: 4616)."""
    env_name = 'Pendulum-v0'
    hidden_sizes = [16]
    output_activation = 'none'
    network = 'policy.lstm_network.LSTMNetwork'

    pop_size = 200
    lr = 0.05
    sigma = 0.07

    num_generations = 700

    env_seed = -1
    num_episodes = 5


@ex.named_config
def pendulum_partial():
    """LSTM on partially observable task, does not work well enough."""
    env_name = 'PartiallyObservablePendulum'
    hidden_sizes = [16]
    output_activation = 'none'
    network = 'policy.lstm_network.LSTMNetwork'

    env_seed = -1
    num_episodes = 5

    num_generations = 700
    pop_size = 50

    lr = 0.02
    sigma = 0.02
    l2coeff = 0.001


@ex.named_config
def cartpole():
    env_name = 'CartPole-v0'  # https://github.com/openai/gym/wiki/CartPole-v0
    hidden_sizes = [5]
    output_activation = 'none'
    network = 'policy.elman_network.ElmanNetwork'

    lr = 0.01
    sigma = 0.02

    num_generations = 1000
    pop_size = 24

    env_seed = -1
    num_episodes = 5

    l2coeff = 0.001
    recycle = False

    mean_fitness = False


@ex.automain
def run(_run, _config):
    config = Namespace(**_config)

    redis_config = {'unix_socket_path': LOCAL_MASTER_SOCKET_PATH if config.local else REMOTE_MASTER_SOCKET_PATH}

    run_head(redis_config, config, _run, sacred_writer)

