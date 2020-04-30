from typing import List, Dict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats.mstats import gmean
from torch import Tensor

from badger_utils.sacred import SacredUtils
from utils.sacred_local import get_sacred_storage

sacred_utils = SacredUtils(get_sacred_storage())


def plot_df_min_max(df: pd.DataFrame, label: str, p=plt, include_min_max: bool = True, use_geometric_mean: bool = False):
    df = df.dropna()
    df_agg = df.agg(['min', 'max'], axis=1)
    df_mean = df.agg(gmean if use_geometric_mean else np.mean, axis=1)
    mean_line = p.plot(df_mean)[0]
    mean_line.set_label(label)
    if include_min_max:
        p.fill_between(x=df_agg.index, y1='min', y2='max', data=df_agg, alpha=0.25)
    p.plot(df, color=mean_line.get_color(), alpha=0.20)


def read_metric(run_id: int, metric: str, average_window: int = 1):
    metrics = sacred_utils.get_reader(run_id).load_metrics(average_window)
    return metrics[filter(lambda x: x.startswith(f'{metric}'), metrics.columns)]


def plot_run_metrics(run_id: int,
                     metrics: List[str],
                     title: str,
                     yscale: str = 'linear',
                     window: int = 1):
    fig = plt.figure(figsize=(8, 4))
    ax = fig.add_subplot(title=title, xlabel='generation', ylabel='fitness', yscale=yscale)
    for metric in metrics:
        data = read_metric(run_id, f'{metric}', average_window=window)
        plot_df_min_max(data, metric, ax)
    ax.legend()
    return fig


def plot_metrics(metrics: Dict[str, Tensor], title: str, yscale: str = 'linear'):
    fig = plt.figure(figsize=(8, 4))
    ax = fig.add_subplot(title=title, xlabel='epoch', ylabel='mse', yscale=yscale)
    for label, data in metrics.items():
        plot_df_min_max(data, label, ax)
    ax.legend()
    return fig