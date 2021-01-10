from envs.brown_inventory_time_env import BrownInventoryTimeStateEnv
from learning.agents import *
from models.brownian_model import StochasticProcess
from plotting import plot_utils


def main():

    n_ep = 1000
    env = BrownInventoryTimeStateEnv(1, 0.005, StochasticProcess(1, 0.005, 2, 100), 4, 1)
    q, stats = q_learning(env, n_ep)
    _, stats2 = zero_tick(env, n_ep)
    _, stats3 = random_actions(env, n_ep)
    print()

    plot_utils.plot_episode_rewards(stats, 50)
    plot_utils.plot_episode_profit(stats)
    plot_utils.plot_episode_inventory(stats)

    plot_utils.plot_value_heatmap(q)

    print("Profit means, Q, Zero, Random")
    print(stats.episode_profits.mean())
    print(stats2.episode_profits.mean())
    print(stats3.episode_profits.mean())
    print("Profit SDs, Q, Zero, Random")
    print(stats.episode_profits.std())
    print(stats2.episode_profits.std())
    print(stats3.episode_profits.std())
    print("Inventory means, Q, Zero, Random")
    print(stats.episode_inventory.mean())
    print(stats2.episode_inventory.mean())
    print(stats3.episode_inventory.mean())
    print("Inventory SDs, Q, Zero, Random")
    print(stats.episode_inventory.std())
    print(stats2.episode_inventory.std())
    print(stats3.episode_inventory.std())

    plot_utils.plot_relative_profits([stats, stats2, stats3], 1)
    plot_utils.plot_relative_invs([stats, stats2, stats3], 1)


if __name__ == '__main__':
    main()
