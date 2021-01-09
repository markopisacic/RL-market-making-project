import numpy as np
import math
import matplotlib.pyplot as plt


class StochasticProcess:
    """
    Brownian motion to simulate stock price changes.
    """

    def __init__(self, total_time, dt, volatility, initial_asset_price, drift=0):
        self.total_time = total_time
        self.dt = dt
        self.volatility = volatility
        self.initial_asset_price = initial_asset_price
        self.current_asset_price = initial_asset_price
        self.asset_prices = [initial_asset_price]
        self.drift = drift

    def generate_series(self):
        """
        Run one price simulation in given time interval, save asset prices to internal state and return them.
        """
        self.current_asset_price = self.initial_asset_price
        self.asset_prices = [self.initial_asset_price]

        for i in range(int(self.total_time / self.dt)):
            # simpler model than standard geometric Brownian motion, to be compatible with Avellaneda-Stoikov work
            # dS = self.current_asset_price * (self.drift * self.dt + self.volatility * dW)
            dS = self.volatility * np.random.normal(0, math.sqrt(self.dt))
            self.current_asset_price += dS
            self.asset_prices.append(self.current_asset_price)

        return self.asset_prices

    def to_file(self, name):
        """
        Save current asset prices to text file.
        """
        np.savetxt(name, self.asset_prices, delimiter=':')

    @staticmethod
    def data_from_file(filename, sep):
        """
        Static method to retrieve an array of prices from text file.
        """
        return np.genfromtxt(fname=filename, delimiter=sep)


if __name__ == "__main__":
    p = StochasticProcess(1, 0.005, 2, 100)
    p.generate_series()
    p.to_file('models/data_new.csv')
    plt.plot(np.arange(0, len(p.asset_prices)), p.asset_prices)
    plt.show()
