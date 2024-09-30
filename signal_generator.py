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
        """
        Generate a sinusoidal signal of the form: a*sin(k*x - omega*t)*exp(gamma*t)

        Parameters
        ----------
        a : float, optional
            Amplitude of the sinusoidal signal, by default 1
        k : float, optional
            Spatial frequency of the signal, by default 0.1
        omega : float, optional
            Temporal frequency of the signal, by default 1
        gamma : float, optional
            Temporal decay rate of the signal, by default 0
        """
        signal = np.sin(k*self.X - omega*self.T)*np.exp(gamma*self.T)
        spatial_norm = np.linalg.norm(signal, axis=-1, ord=2)
        signal = signal / spatial_norm[:, None]
        signal = a * signal
        my_dict = {'type': 'sinusoid1', 'a': a, 'k': k, 'omega': omega, 'gamma': gamma, 'signal': signal}
        self.components.append(my_dict)
        self.signal += signal

    def add_sinusoid2(self, a=1, k=0.2, omega=1, c=0):
        """
        Generate a sinusoidal signal of the form: a*(exp(-k*(x+c)^2)*cos(omega*t)

        Parameters
        ----------
        a : float, optional
            Amplitude (area under the curve) of the signal, by default 1
        k : float, optional
            Spatial exponential decay rate of the signal, by default 0.2
        omega : float, optional
            Temporal frequency of the signal, by default 1
        c : float, optional
            Offset of the signal, by default 0
        """
        spatial_signal = np.exp(-k*(self.X+c)**2)
        area = np.trapz(spatial_signal, self.x, axis=-1)[0]  # Compute the area under the curve
        signal = a * spatial_signal / area * np.cos(omega*self.T)
        my_dict = {'type': 'sinusoid2', 'a': a, 'k': k, 'omega': omega, 'c': c, 'signal': signal}
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


def sample_data(data, t, dt=1, duration=None):
    """
    Given an input spatio-temporal dataset,
    sample the data at a given temporal rate.

    Parameters
    ----------
    data : np.ndarray
        Input spatio-temporal data with shape (time, space)
    t : np.ndarray
        Corresponding time vector with shape (time,)
    dt : int, optional
        Temporal sampling rate, i.e. how many time steps to skip.
        By default, 1.
    duration : int, optional
        Number of time steps to sample. If None, sample the entire dataset.
        By default, None.
    """
    if duration is None:
        duration = data.shape[0]

    # Sample the data
    data_sampled = data[:duration:dt]
    t_sampled = t[:duration:dt]

    return data_sampled, t_sampled
