from collections import namedtuple

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt

EpisodeStats = namedtuple("Stats", ["episode_lengths", "episode_rewards", "episode_profits", "episode_inventory"])


def plot_value_heatmap(q):
    """For every state-action pair, show action value. Results are shown in a heatmap."""
    ser = pd.DataFrame.from_dict(dict(q), orient='index')
    ser = ser.sort_index()
    plt.figure(figsize=(20, 10))
    sns.heatmap(ser, linewidths=0.5, cmap="YlGnBu", square=False)
    plt.show()


def plot_episode_lengths(stats):
    """
    Shows episode length through time
    :param stats: EpisodeStats
    :return: figure
    """
    # Plot the episode length over time
    fig = plt.figure(figsize=(10, 5))
    plt.plot(stats.episode_lengths)
    plt.xlabel("Episode")
    plt.ylabel("Episode Length")
    plt.title("Episode Length over Time")
    plt.show()
    return fig


def plot_episode_rewards(stats, smoothing_window=10):
    """
    Shows reward amounts through time (smoothed).
    :param stats: EpisodeStats
    :param smoothing_window: smoothing window
    :return: figure
    """
    fig = plt.figure(figsize=(10, 5))
    rewards_smoothed = pd.Series(stats.episode_rewards).rolling(smoothing_window, min_periods=smoothing_window).mean()
    plt.plot(rewards_smoothed)
    plt.xlabel("Episode")
    plt.ylabel("Episode Reward (Smoothed)")
    plt.title("Episode Reward over Time (Smoothed over window size {})".format(smoothing_window))
    plt.show()
    return fig


def plot_episode_profit(stats):
    """
    Shows profit through time
    :param stats: EpisodeStats
    :return: figure
    """
    fig = plt.figure(figsize=(10, 5))
    plt.plot(stats.episode_profits)
    plt.xlabel("Episode")
    plt.ylabel("Episode profit")
    plt.title("Episode Profit over Time")
    plt.show()
    return fig


def plot_episode_inventory(stats):
    """
    Shows inventory at the end of every episode.
    :param stats: EpisodeStats
    :return: figure
    """
    fig = plt.figure(figsize=(10, 5))
    plt.plot(stats.episode_inventory)
    plt.xlabel("Episode")
    plt.ylabel("Episode Inventory")
    plt.title("Episode Inventory over Time")
    plt.show()
    return fig


def plot_episode_times(stats):
    """
    Shows number of episodes completed through time.
    :param stats: EpisodeStats
    :return: figure
    """
    fig = plt.figure(figsize=(10, 5))
    plt.plot(np.cumsum(stats.episode_lengths), np.arange(len(stats.episode_lengths)))
    plt.xlabel("Time Steps")
    plt.ylabel("Episode")
    plt.title("Episode per time step")
    plt.show()
    return fig
