import numpy as np


class SignalGenerator:

    def __init__(
            self,
            nx=100,
            nt=500,
            x_min=0,
            x_max=10,
            t_min=0,
            t_max=50,
    ):
        self.x = np.linspace(x_min, x_max, nx)
        self.t = np.linspace(t_min, t_max, nt)
        self.X, self.T = np.meshgrid(self.x, self.t)
        self.signal = np.zeros(self.X.shape)
        self.components = []

    def add_sinusoid1(self, a=1, k=0.1, omega=1, gamma=0):
        signal = a * np.sin(k*self.X - omega*self.T)*np.exp(gamma*self.T)
        my_dict = {'type': 'sinusoid1', 'a': a, 'k': k, 'omega': omega, 'gamma': gamma, 'signal': signal}
        self.components.append(my_dict)
        self.signal += signal

    def add_sinusoid2(self, a=1, omega=1):
        signal = a * np.exp(-0.2*self.X*self.X) * np.cos(omega*self.T)
        my_dict = {'type': 'sinusoid2', 'a': a, 'omega': omega, 'signal': signal}
        self.components.append(my_dict)
        self.signal += signal

    def add_sinusoid3(self, a=1, omega=1):
        signal = a * 1.0/np.cosh(0.5*(self.X + 2)) * np.cos(omega*self.T)
        my_dict = {'type': 'sinusoid3', 'a': a, 'omega': omega, 'signal': signal}
        self.components.append(my_dict)
        self.signal += signal

    def add_trend(self, mu=0.2, trend=0.01):
        signal = self.T*trend + mu
        my_dict = {'type': 'trend', 'mu': mu, 'trend': trend, 'signal': signal}
        self.components.append(my_dict)
        self.signal += signal

    def add_noise(self, noise_std=0.1, random_seed=None):
        np.random.seed(random_seed)
        self.signal += np.random.normal(0, noise_std, self.signal.shape)
