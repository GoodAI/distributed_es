from typing import List

import click
import matplotlib.pyplot as plt
import numpy as np

from blogpost.metrics import read_metric

FOLDER = 'blogpost/img/'


@click.group()
def cli():
    pass


def smooth_old(data: np.ndarray, window_size: int):
    return np.convolve(data, np.ones((window_size,)) / window_size, mode='valid')


def smooth(data: np.ndarray, window_size: int):
    """1D convolution over the data,
     compared to np.convolve, this adaptively increases the size of the window at the beginning,
     to preserve details where needed"""
    result = np.zeros(data.size - window_size)
    window = np.ones((window_size, ))

    for pos in range(result.size):
        win_size = min(pos + 1, window_size)

        if win_size < window_size:
            win = np.ones((win_size, ))
        else:
            win = window

        result[pos] = np.sum(data[pos + 1 - win_size:pos + 1] * win) / win.size
    return result


def _resize(all_lines: List[np.ndarray], run_ids: List[int]) -> List[np.ndarray]:

    sizes = [line.size for line in all_lines]
    sizes_differ = len(set(sizes)) != 1

    if sizes_differ:
        min_size = min(sizes)
        print(f'WARNING: lines from these experiments have different lengths: {run_ids}, will crop to the smallest')
        all_lines = [line[:min_size] for line in all_lines]

    return all_lines


def plot_runs(run_ids: List[int],
              label: str,
              metrics: List[str] = ['current fitness'],
              window: int = 20,
              include_min_max: bool = True,
              include_all: bool = False):
    all_lines = []

    for run_id in run_ids:
        print(f'Reading metric from {run_id}..')
        data = read_metric(run_id, f'{metrics[0]}').to_numpy()
        assert data.shape[1] == 1
        all_lines.append(data.reshape(-1))

    print(f'Data loaded')
    all_lines = _resize(all_lines, run_ids)
    all_lines = [smooth(line, window) for line in all_lines]
    result = np.stack(all_lines)

    min_vals = np.min(result, axis=0)
    max_vals = np.max(result, axis=0)
    mean_vals = np.mean(result, axis=0)
    gens = np.arange(mean_vals.size)

    mean_line = plt.plot(mean_vals, alpha=0.01)[0]
    if include_min_max:
        plt.fill_between(x=gens, y1=min_vals, y2=max_vals, alpha=0.25)
    if include_all:
        for line in result:
            plt.plot(line, color=mean_line.get_color(), alpha=0.6, linewidth=0.5)
    mean_line = plt.plot(mean_vals, color=mean_line.get_color(), alpha=0.8)[0]
    mean_line.set_label(label)

    print(f'done, loaded {len(run_ids)} of series under label: {label}')


@cli.command()
def lander():
    """Plot results of the lander from the Sacred"""

    lander_es_one_ep = [5024, 5025, 5026, 5027, 5028, 5029]
    lander_es_five_ep = [5030, 5031, 5032, 5033, 5034, 5035]
    lander_cma_one_ep = [5036, 5037, 5038, 5039, 5040, 5040, 5041]
    lander_cma_five_ep = [5042, 5043, 5044, 5045, 5046, 5047, 5048]

    lander_es_one_ep_20 = [5049, 5050, 5051, 5052, 5053, 5054]
    lander_es_five_ep_20 = [5055, 5056, 5057, 5058, 5059, 5060]
    lander_es_one_ep_200 = [5061, 5062, 5063, 5064, 5065, 5066]

    fig = plt.figure(figsize=(8, 5))
    ax = fig.add_subplot(title='LunarLanderContinuous; FF policy',
                         xlabel='Generation',
                         ylabel='Fitness value',
                         yscale='linear')

    plot_runs(lander_es_one_ep, label='ES; 1ep')
    plot_runs(lander_es_five_ep, label='ES; 5eps')
    plot_runs(lander_cma_one_ep, label='CMA; 1ep')
    plot_runs(lander_cma_five_ep, label='CMA; 5eps')

    plot_runs(lander_es_one_ep_20, label='OptimES; 1ep; pop_size=20')
    plot_runs(lander_es_five_ep_20, label='OptimES; 5eps; pop_size=20')
    plot_runs(lander_es_one_ep_200, label='OptimES; 1ep; pop_size=200')

    plt.legend()
    plt.ylim(bottom=-700)

    plt.savefig(f'{FOLDER}/lunar_lander_continuous.png', format='png')
    plt.show()


@cli.command()
def ant():
    """Plot results of the ant from the Sacred"""
    x = [5070, 5070, 5072, 5073, 5074, 5075]  # old run
    es_pop_size100_1episode = [5087, 5086, 5085, 5084, 5083]
    cma = [5177, 5179, 5180, 5181, 5183]
    es = [5196, 5197, 5198, 5199, 5200]

    cma_5ep = [5223, 5224, 5225]  # 5222, shorter run
    es_5ep = [5214, 5218, 5219]  # 5217, shorter run

    fig = plt.figure(figsize=(8, 5))
    ax = fig.add_subplot(title='AntPyBulletEnv-v0; FF policy',
                         xlabel='Generation',
                         ylabel='Fitness value',
                         yscale='linear')

    plot_runs(es, label='ES; 1ep', include_all=True)
    plot_runs(es_5ep, label='ES; 5ep', include_all=True)
    plot_runs(cma, label='CMA, 1ep', include_all=True)
    plot_runs(cma_5ep, label='CMA, 5ep', include_all=True)
    plot_runs(es_pop_size100_1episode, label='OptimES; 1ep, pop_size=100')
    plt.legend(loc='lower right')

    plt.savefig(f'{FOLDER}/ant.png', format='png')
    plt.show()


@cli.command()
def pendulum():
    """Plot results of the partially observable pendulum from the Sacred"""

    cma = [5160, 5162, 5163, 5165, 5167]
    cma_30ep = [5231, 5232, 5234, 5235, 5238]
    optim_es = [5122, 5123, 5124, 5126, 5127, 5129]
    optim_es_pop_size200 = [5154, 5156, 5157, 5158, 5159]
    es = [5114, 5115, 5117, 5118, 5119]

    fig = plt.figure(figsize=(8, 5))
    ax = fig.add_subplot(title='PartiallyObservablePendulum; LSTM policy',
                         xlabel='Generation',
                         ylabel='Fitness value',
                         yscale='linear')

    plot_runs(es, label='ES; 5ep')
    plot_runs(cma, label='CMA; 5ep', include_all=True)
    plot_runs(cma_30ep, label='CMA; 30ep', include_all=True)
    plot_runs(optim_es, label='OptimES; 10ep, pop_size=50')
    plot_runs(optim_es_pop_size200, label='OptimES; 5ep, pop_size=200')
    plt.legend()
    plt.ylim(bottom=-1400)
    # plt.ylim(top=-259)

    plt.savefig(f'{FOLDER}/pendulum_partial.png', format='png')
    plt.show()


if __name__ == '__main__':
    """This script was used to generate graphs for the blogpost/README.md"""
    cli()
