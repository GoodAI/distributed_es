
AUTHOR = 'unknown'

default_common_params = {

    'author': AUTHOR,

    # -------- Policy
    'hidden_sizes': [64, 64],
    'network': 'policy.ff_network.FFNetwork',
    'hidden_activation': 'tanh',
    'output_activation': 'discrete',

    # -------- Env
    'env_name': 'LunarLanderContinuous-v2',

    # -------- Simulate
    'num_episodes': 1,  # how many epochs to run per fitness eval?
    'env_seed': -1,  # None=constant per generation, -1=random every the time
    'max_ep_length': None,  # optional
    'load_from': None,  # deserialize the policy from a given exp_id

    # -------- ES - common
    'pop_size': 50,
    'num_generations': 1000,

    # -------- Other
    'num_serializations': 10,  # how many times to serialize the model during given num_generations?
    'force_cpu': False,  # use CPU for PyTorch?

    'render': False,  # render?
    'sleep': 0.01,  # sleep for rendering
}

default_es_params = {
    'lr': 0.005,
    'l2coeff': 0.005,
    'sigma': 0.02,

    'adam_beta1': 0.9,
    'adam_beta2': 0.999,
    'adam_epsilon': 1e-08,

    'local': True,  # tells the (OpenAI's ES) experiment if is ran locally or on the cluster
    'min_task_runtime': 0.01,  # run evaluation at least for this time
}

default_nevergrad_params = {
    'tool': 'CMA',
    'use_ray': True,
}
