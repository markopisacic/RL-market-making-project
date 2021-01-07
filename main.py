from envs.brown_inventory_time_env import BrownInventoryTimeStateEnv
from learning.q_learning import q_learning
from models.brownian_model import StochasticProcess
from plotting import plot_utils


def main():
    env = BrownInventoryTimeStateEnv(12 * 0.001, 0.001, StochasticProcess(.2, .3, 0.001, 100), 1, 1)
    q, stats = q_learning(env, 1000)
    print()

    plot_utils.plot_episode_rewards(stats, 50)
    plot_utils.plot_episode_profit(stats)
    plot_utils.plot_episode_inventory(stats)
    plot_utils.plot_value_heatmap(q)


if __name__ == '__main__':
    main()
