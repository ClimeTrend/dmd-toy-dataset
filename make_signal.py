import numpy as np


def f1slow(x, t, L):
    return np.cos(2*np.pi*x/(2*L)) * np.cos(0.3 * t)


def f2slow(x, t):
    return np.exp(-0.2*x*x) * np.cos(0.2 * t)


def f1med(x, t):
    return 1.0 / np.cosh(0.5*(x + 2)) * np.cos(1.3 * t)


def f2med(x, t):
    return 2.0 / np.cosh(0.2*x) * np.tanh(0.2*x) * np.sin(0.8 * t)


def f1fast(x, t):
    return np.exp(-(x - 1)*(x-1)) * np.cos(5.3 * t)


def f2fast(x, t):
    return x*x*np.exp(-(x + 1)*(x+1)) * np.cos(6.0 * t+np.pi/4)


def make_signal(
        signals="all", noise=True, noise_std=0.5, nx=200, nt=400, L=5., T=4.
        ):
    """
    Generate a signal as a superposition of two components.
    Each component is chosen from a set of three slow, medium, and fast
    signals. The signal is defined on a 2D grid of size nx x nt.
    The spatial domain is [-L, L] and the temporal domain is [0, T*pi].
    The signal is corrupted by Gaussian noise with standard deviation noise_std.

    Parameters
    ----------
    signals : str
        Type of signals to include in the superposition.
        Possible values are 'all', 'slow', 'med', and 'fast'.
        Default is 'all' (all three types).
    noise : bool
        Whether to add Gaussian noise to the signal.
        Default is True.
    noise_std : float
        Standard deviation of the Gaussian noise.
        Default is 0.5.
    nx : int
        Number of spatial points.
        Default is 200.
    nt : int
        Number of temporal points.
        Default is 400.
    L : float
        Spatial domain is [-L, L].
        Default is 5.
    T : float
        Temporal domain is [0, T*pi].
        Default is 4.

    Returns
    -------
    x : 1D array
        The spatial grid.
    t : 1D array
        The temporal grid.
    f : 2D array
        The signal as a superposition of f1 and f2.
    f1 : 2D array
        The first component of the signal.
    f2 : 2D array
        The second component of the signal.
    """

    x = np.linspace(-L, L, nx)
    t = np.linspace(0, T*np.pi, nt)
    X, T = np.meshgrid(x, t)

    if signals == "all":
        f1 = f1slow(X, T, L) + f1med(X, T) + f1fast(X, T)
        f2 = f2slow(X, T) + f2med(X, T) + f2fast(X, T)
    elif signals == "slow":
        f1 = f1slow(X, T, L)
        f2 = f2slow(X, T)
    elif signals == "med":
        f1 = f1med(X, T)
        f2 = f2med(X, T)
    elif signals == "fast":
        f1 = f1fast(X, T)
        f2 = f2fast(X, T)
    else:
        raise ValueError("Unknown signal type")
    f = f1 + f2

    if noise:
        f += np.random.normal(0, noise_std, f.shape)

    return x, t, f, f1, f2
