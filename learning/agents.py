import itertools
import sys
from collections import defaultdict

import numpy as np

from plotting.plot_utils import EpisodeStats


def make_epsilon_greedy_policy(Q, epsilon, nA):
    """
    Creates an epsilon-greedy policy based on a given Q-function and epsilon.
    """

    def policy_fn(observation):
        A = np.ones(nA, dtype=float) * epsilon / nA
        best_action = np.argmax(Q[observation])
        A[best_action] += (1.0 - epsilon)
        return A

    return policy_fn


def q_learning(env, num_episodes, discount_factor=1.0, alpha=0.5, epsilon=0.1):
    """
    Q-Learning algorithm: Off-policy TD control. Finds the optimal greedy policy
    while following an epsilon-greedy policy
    """

    # A nested dictionary that maps state -> (action -> action-value).
    Q = defaultdict(lambda: np.zeros(env.action_space_size))

    # Keeps track of useful statistics
    stats = EpisodeStats("Q-learning",
                         episode_lengths=np.zeros(num_episodes),
                         episode_rewards=np.zeros(num_episodes),
                         episode_profits=np.zeros(num_episodes),
                         episode_inventory=np.zeros(num_episodes))

    # The policy we're following
    policy = make_epsilon_greedy_policy(Q, epsilon, env.action_space_size)

    for i_episode in range(num_episodes):
        if (i_episode + 1) % 10 == 0:
            print("\rEpisode {}/{}.".format(i_episode + 1, num_episodes), end="")
            sys.stdout.flush()

        # Reset the environment and pick the first action
        state = env.reset()

        # Step through the environment until finished
        for t in itertools.count():

            # Take a step
            action_probs = policy(state)
            action = np.random.choice(np.arange(len(action_probs)), p=action_probs)
            next_state, reward, done, w, i = env.step(action)

            # Update statistics
            stats.episode_rewards[i_episode] += reward
            stats.episode_lengths[i_episode] = t
            stats.episode_profits[i_episode] = w
            stats.episode_inventory[i_episode] = i

            # TD Update
            best_next_action = np.argmax(Q[next_state])
            td_target = reward + discount_factor * Q[next_state][best_next_action]
            td_delta = td_target - Q[state][action]
            Q[state][action] += alpha * td_delta

            if done:
                break

            state = next_state

    return Q, stats


def random_actions(env, num_episodes):
    """
    This agent takes random actions for each step of the way. No learning is present, so returning value function
    results in None.
    """

    # Keeps track of useful statistics
    stats = EpisodeStats("Random actions",
                         episode_lengths=np.zeros(num_episodes),
                         episode_rewards=np.zeros(num_episodes),
                         episode_profits=np.zeros(num_episodes),
                         episode_inventory=np.zeros(num_episodes))

    for i_episode in range(num_episodes):
        if (i_episode + 1) % 10 == 0:
            print("\rEpisode {}/{}.".format(i_episode + 1, num_episodes), end="")
            sys.stdout.flush()

        # Reset the environment and pick the first action
        env.reset()

        # Step through the environment until finished
        for t in itertools.count():

            # Take a random step
            action = np.random.choice(np.arange(env.action_space_size))
            next_state, reward, done, w, i = env.step(action)

            # Update statistics
            stats.episode_rewards[i_episode] += reward
            stats.episode_lengths[i_episode] = t
            stats.episode_profits[i_episode] = w
            stats.episode_inventory[i_episode] = i

            if done:
                break

    return None, stats


def zero_tick(env, num_episodes):
    """
    This agent will always pick action that is 0 ticks away from best id and best ask price.  No learning is present,
    so returning value function results in None.
    """

    # Keeps track of useful statistics
    stats = EpisodeStats("Zero-tick",
                         episode_lengths=np.zeros(num_episodes),
                         episode_rewards=np.zeros(num_episodes),
                         episode_profits=np.zeros(num_episodes),
                         episode_inventory=np.zeros(num_episodes))

    for i_episode in range(num_episodes):
        if (i_episode + 1) % 10 == 0:
            print("\rEpisode {}/{}.".format(i_episode + 1, num_episodes), end="")
            sys.stdout.flush()

        # Reset the environment and pick the first action
        env.reset()

        # Step through the environment until finished
        for t in itertools.count():

            action = 0
            next_state, reward, done, w, i = env.step(action)

            # Update statistics
            stats.episode_rewards[i_episode] += reward
            stats.episode_lengths[i_episode] = t
            stats.episode_profits[i_episode] = w
            stats.episode_inventory[i_episode] += abs(i)

            if done:
                stats.episode_inventory[i_episode] /= t
                break

    return None, stats
