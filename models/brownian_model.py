import numpy as np
import math
import matplotlib.pyplot as plt


class StochasticProcess:
    """
    Brownian motion to simulate stock price changes.
    """

    def __init__(self, drift, volatility, delta_t, initial_asset_price):
        self.drift = drift
        self.volatility = volatility
        self.delta_t = delta_t
        self.initial_asset_price = initial_asset_price
        self.current_asset_price = initial_asset_price
        self.asset_prices = [initial_asset_price]

    def time_step(self):
        dW = np.random.normal(0, math.sqrt(self.delta_t))
        dS = self.current_asset_price * (self.drift * self.delta_t + self.volatility * dW)
        self.current_asset_price += dS
        self.asset_prices.append(self.current_asset_price)

    def generate_series(self, delta_t, k_timesteps):
        self.current_asset_price = self.initial_asset_price
        self.asset_prices = [self.initial_asset_price]

        tte = k_timesteps * delta_t
        while (tte - delta_t) > 0:
            self.time_step()
            tte = tte - delta_t
        return self.asset_prices

    def to_file(self, name):
        np.savetxt(name, self.asset_prices, delimiter=':')

    @staticmethod
    def data_from_file(filename, sep):
        return np.genfromtxt(fname=filename, delimiter=sep)


if __name__ == "__main__":
    delta_t = 0.001
    k_timesteps = 12
    p = StochasticProcess(.2, .3, delta_t, 100)
    p.generate_series(delta_t, k_timesteps)
    p.to_file('data_new.csv')
    plt.plot(np.arange(0, len(p.asset_prices)), p.asset_prices)
    plt.show()
