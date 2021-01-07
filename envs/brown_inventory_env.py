import math
import random

from envs.base_env import BaseEnv


class BrownInventoryStateEnv(BaseEnv):
    """
    This environment uses Brownian motion to model stock price.
    Its states are defined solely by inventory states, which are binned to reduce space size.
    """

    def __init__(self, total_time, delta_t, process, a, b):
        super(BrownInventoryStateEnv, self).__init__()

        # Parameters used in reward function
        self.a = a
        self.b = b

        # Parameters related to price
        self.tick = 10
        self.value = 1000
        self.inventory = 0

        # Time parameters
        self.total_time = total_time
        self.current_time = 0
        self.delta_t = delta_t
        self.k_timesteps = total_time / delta_t

        # Action and observation space
        self.action_space = 7
        self.observation_space = 9

        # Brownian stock price simulation data
        self.process = process
        self.data = self.process.generate_series(self.delta_t, self.k_timesteps)
        self.iteration = 0
        self.current_price = self.data[self.iteration]

    def step(self, action):

        # Previous values needed for reward calculation
        prev_inventory = self.inventory
        prev_wealth = self._determine_wealth()

        # Get distance (in ticks) from action
        d_bid = action // 3
        d_ask = action % 3

        # Based on tick distance, calculate bid/ask execution probability
        # TODO ovaj dio je vjerojatno najlosije implementiran (najmanje u skladu s paperom) pa je prvi kandidat za promjenu
        p_factor = 0.66
        p_bid = math.exp(-d_bid) * p_factor
        p_ask = math.exp(-d_ask) * p_factor
        if random.random() < p_bid:
            self.value -= (self.current_price - d_bid * self.tick)
            self.inventory += 1
        if random.random() < p_ask:
            self.value += (self.current_price + d_ask * self.tick)
            self.inventory -= 1

        # Determine new state, reward
        next_state = self._determine_state()
        reward = self.a * (self._determine_wealth() - prev_wealth) + \
                 math.copysign(math.exp(self.b * (self.total_time - self.current_time)),
                               abs(self.inventory) - abs(prev_inventory)
                               )
        done = self.total_time - self.delta_t <= self.current_time

        # Increment counters
        self.current_time += self.delta_t
        self.iteration += 1
        if not done:
            self.current_price = self.data[self.iteration]

        return next_state, reward, done, self._determine_wealth(), self.inventory

    def reset(self):
        self.value = 1000
        self.inventory = 0

        self.current_time = 0

        self.data = self.process.generate_series(self.delta_t, self.k_timesteps)
        self.iteration = 0
        self.current_price = self.data[self.iteration]

        return self._determine_state()

    def _determine_wealth(self):
        """
        Calculate wealth using current money, inventory, and stock price.
        :return: Current wealth
        """
        return self.value + self.inventory * self.current_price

    def _determine_state(self):
        """
        State is determined only by inventory value. Values are binned into a smaller number of groups.
        :return: Current state
        """
        if self.inventory < -4:
            return 6
        elif -4 <= self.inventory < -2:
            return 5
        elif -2 <= self.inventory < 0:
            return 4
        elif self.inventory > 4:
            return 3
        elif 2 < self.inventory <= 4:
            return 2
        elif 0 < self.inventory <= 2:
            return 1
        else:
            return 0
