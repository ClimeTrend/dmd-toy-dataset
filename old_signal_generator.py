import numpy as np


class SignalGenerator:
    """
    SignalGenerator class generates synthetic signals composed of slow, medium, and fast varying components.

    Attributes:
        nx (int): Number of spatial points. Default is 200.
        nt (int): Number of temporal points. Default is 400.
        x_length (float): Spatial domain length. Default is 5.
        t_length (float): Temporal domain length. Default is 4.
        x (ndarray): Spatial points.
        t (ndarray): Temporal points.
        X (ndarray): Meshgrid of spatial points.
        T (ndarray): Meshgrid of temporal points.

        omega_f1slow (float): Frequency for the first slow component. Default is 0.3 (rad/s).
        omega_f2slow (float): Frequency for the second slow component. Default is 0.2 (rad/s).
        omega_f1med (float): Frequency for the first medium component. Default is 1.3 (rad/s).
        omega_f2med (float): Frequency for the second medium component. Default is 0.8 (rad/s).
        omega_f1fast (float): Frequency for the first fast component. Default is 5.3 (rad/s).
        omega_f2fast (float): Frequency for the second fast component. Default is 6.0 (rad/s).

        f1_slow (ndarray): First slow varying component.
        f2_slow (ndarray): Second slow varying component.
        f1_med (ndarray): First medium varying component.
        f2_med (ndarray): Second medium varying component.
        f1_fast (ndarray): First fast varying component.
        f2_fast (ndarray): Second fast varying component.
        f (ndarray): Combined signal, initialized as zeros.
            Update using add_signal method, and add noise using add_noise method.

    Methods:
        add_signal: Adds the specified signal type to the combined signal.
            Args:
                which (str): Signal type to add. Options are "slow", "med", "fast" and "all".
        add_noise: Adds Gaussian noise to the combined signal.
            Args:
                noise_std (float): Standard deviation of the Gaussian noise. Default is 0.25.
                random_seed (int): Random seed for reproducibility. Default is None.
    """
    def __init__(
            self,
            nx=200,
            nt=400,
            x_length=5.,
            t_length=4.,
            omega_f1slow=0.3,
            omega_f2slow=0.2,
            omega_f1med=1.3,
            omega_f2med=1.0,
            omega_f1fast=5.3,
            omega_f2fast=6.0
            ):
        self.nx = nx
        self.nt = nt
        self.x_length = x_length
        self.t_length = t_length
        self.x = np.linspace(-x_length, x_length, nx)
        self.t = np.linspace(0, t_length*np.pi, nt)
        self.X, self.T = np.meshgrid(self.x, self.t)
        self.omega_f1slow = omega_f1slow
        self.omega_f2slow = omega_f2slow
        self.omega_f1med = omega_f1med
        self.omega_f2med = omega_f2med
        self.omega_f1fast = omega_f1fast
        self.omega_f2fast = omega_f2fast
        self.f1_slow = self._f1slow(self.X, self.T, self.omega_f1slow)
        self.f2_slow = self._f2slow(self.X, self.T, self.omega_f2slow)
        self.f1_med = self._f1med(self.X, self.T, self.omega_f1med)
        self.f2_med = self._f2med(self.X, self.T, self.omega_f2med)
        self.f1_fast = self._f1fast(self.X, self.T, self.omega_f1fast)
        self.f2_fast = self._f2fast(self.X, self.T, self.omega_f2fast)
        self.f = np.zeros(self.f1_slow.shape)

    def _f1slow(self, x, t, omega):
        f = np.cos(2*np.pi*x/(2*self.x_length)) * np.cos(omega*t)
        return f

    def _f2slow(self, x, t, omega):
        f = np.exp(-0.2*x*x) * np.cos(omega*t)
        return f

    def _f1med(self, x, t, omega):
        f = 1.0/np.cosh(0.5*(x + 2)) * np.cos(omega*t)
        return f

    def _f2med(self, x, t, omega):
        f = 2.0/np.cosh(0.2*x) * np.tanh(0.2*x) * np.sin(omega*t)
        return f

    def _f1fast(self, x, t, omega):
        f = np.exp(-(x - 1)*(x-1)) * np.cos(omega*t)
        return f

    def _f2fast(self, x, t, omega):
        f = x*x*np.exp(-(x + 1)*(x+1)) * np.cos(omega*t+np.pi/4)
        return f

    def add_signal(self, which):
        if which == "slow":
            self.f += self.f1_slow + self.f2_slow
        elif which == "med":
            self.f += self.f1_med + self.f2_med
        elif which == "fast":
            self.f += self.f1_fast + self.f2_fast
        elif which == "all":
            self.f += self.f1_slow + self.f2_slow + self.f1_med + self.f2_med + self.f1_fast + self.f2_fast
        else:
            raise ValueError("Unknown signal type")

    def add_noise(self, noise_std=0.25, random_seed=None):
        np.random.seed(random_seed)
        self.f += np.random.normal(0, noise_std, self.f.shape)
