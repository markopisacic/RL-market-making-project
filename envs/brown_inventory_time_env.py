import math
import random

from envs.base_env import BaseEnv


class BrownInventoryTimeStateEnv(BaseEnv):
    """
    This environment uses Brownian motion to model stock price.
    Its states are defined by inventory state and time state.
    """

    def __init__(self, total_time, delta_t, process, a, b):
        super(BrownInventoryTimeStateEnv, self).__init__()

        # Parameters used in reward function
        self.a = a
        self.b = b

        # Parameters related to price
        self.tick = 0.1
        self.value = 1000
        self.inventory = 0

        # Time parameters
        self.total_time = total_time
        self.current_time = 0
        self.delta_t = delta_t
        self.time_left = self.total_time / self.delta_t

        # To prevent state number explosion due to high number of temporal states
        self.bin_size = 20

        # Action and observation space
        self.observation_space_size = 7 * ((self.total_time / self.delta_t) // self.bin_size)
        self.action_space_size = 9

        # Brownian stock price simulation data
        self.process = process
        self.data = self.process.generate_series()
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
        A = 140
        k = -1.5
        p_bid = A * math.exp(-k * d_bid) * self.delta_t
        p_ask = A * math.exp(-k * d_ask) * self.delta_t
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

        # print(str(self.iteration) + ", r = " + str(reward) + ", l = " + str(
        #     self.a * (self._determine_wealth() - prev_wealth)) + ", r = " + str(
        #     math.copysign(math.exp(self.b * (self.total_time - self.current_time)),
        #                   abs(prev_inventory) - abs(self.inventory)
        #                   )) + ", (" + str(d_bid) + ", " + str(d_ask) + "), i = " + str(
        #     self.inventory) + ", pi = " + str(prev_inventory))

        # Increment counters
        self.current_time += self.delta_t
        self.iteration += 1
        if not done:
            self.current_price = self.data[self.iteration]
        if done:
            # TODO implement CARA utility
            # this is just a temporary basic implementation
            r = 0.1
            new_reward = math.exp(-r * abs(self.inventory))
            reward = reward * new_reward

        self.time_left -= 1
        return next_state, reward, done, self._determine_wealth(), self.inventory

    def reset(self):
        self.value = 1000
        self.inventory = 0

        self.current_time = 0
        self.time_left = self.total_time / self.delta_t

        self.data = self.process.generate_series()
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
        State is determined by inventory category and remaining time.
        :return: Current state
        """
        time_component = self.time_left // self.bin_size * 7
        if self.inventory < -4:
            inventory_component = 6
        elif -4 <= self.inventory < -2:
            inventory_component = 5
        elif -2 <= self.inventory < 0:
            inventory_component = 4
        elif self.inventory > 4:
            inventory_component = 3
        elif 2 < self.inventory <= 4:
            inventory_component = 2
        elif 0 < self.inventory <= 2:
            inventory_component = 1
        else:
            inventory_component = 0
        # print(self.time_left // self.bins * 7)
        return time_component + inventory_component
