import numpy as np


def f1slow(x, t, L):
    omega = 0.3
    f = np.cos(2*np.pi*x/(2*L)) * np.cos(omega * t)
    return f, omega


def f2slow(x, t):
    omega = 0.2
    f = np.exp(-0.2*x*x) * np.cos(omega * t)
    return f, omega


def f1med(x, t):
    omega = 1.3
    f = 1.0 / np.cosh(0.5*(x + 2)) * np.cos(omega * t)
    return f, omega


def f2med(x, t):
    omega = 0.8
    f = 2.0 / np.cosh(0.2*x) * np.tanh(0.2*x) * np.sin(omega * t)
    return f, omega


def f1fast(x, t):
    omega = 5.3
    f = np.exp(-(x - 1)*(x-1)) * np.cos(omega * t)
    return f, omega


def f2fast(x, t):
    omega = 6.0
    f = x*x*np.exp(-(x + 1)*(x+1)) * np.cos(omega * t+np.pi/4)
    return f, omega


def make_signal(
        signals="all", noise=True, noise_std=0.5, nx=200, nt=400, L=5., T=4.
        ):
    """
    Generate a spatio-temporal signal as a superposition components with different speeds
    (slow, medium, fast). The signal is defined on a 2D grid of size nx x nt.
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
    f: dict
        Dictionary containing the signal components and the superposition.
    x : 1D array
        The spatial grid.
    t : 1D array
        The temporal grid.
    """

    x = np.linspace(-L, L, nx)
    t = np.linspace(0, T*np.pi, nt)
    X, T = np.meshgrid(x, t)

    if signals == "all":
        f = dict.fromkeys(["f1slow", "f2slow", "f1med", "f2med", "f1fast", "f2fast", "f"])
        omegas = dict.fromkeys(["f1slow", "f2slow", "f1med", "f2med", "f1fast", "f2fast"])
        f["f1slow"], omegas["f1slow"] = f1slow(X, T, L)
        f["f2slow"], omegas["f2slow"] = f2slow(X, T)
        f["f1med"], omegas["f1med"] = f1med(X, T)
        f["f2med"], omegas["f2med"] = f2med(X, T)
        f["f1fast"], omegas["f1fast"] = f1fast(X, T)
        f["f2fast"], omegas["f2fast"] = f2fast(X, T)
        f["f"] = f["f1slow"] + f["f2slow"] + f["f1med"] + f["f2med"] + f["f1fast"] + f["f2fast"]
    elif signals == "slow":
        f = dict.fromkeys(["f1slow", "f2slow", "f"])
        omegas = dict.fromkeys(["f1slow", "f2slow"])
        f["f1slow"], omegas["f1slow"] = f1slow(X, T, L)
        f["f2slow"], omegas["f2slow"] = f2slow(X, T)
        f["f"] = f["f1slow"] + f["f2slow"]
    elif signals == "med":
        f = dict.fromkeys(["f1med", "f2med", "f"])
        omegas = dict.fromkeys(["f1med", "f2med"])
        f["f1med"] = f1med(X, T)
        f["f2med"] = f2med(X, T)
        f["f"] = f["f1med"] + f["f2med"]
    elif signals == "fast":
        f = dict.fromkeys(["f1fast", "f2fast", "f"])
        omegas = dict.fromkeys(["f1fast", "f2fast"])
        f["f1fast"], omegas["f1fast"] = f1fast(X, T)
        f["f2fast"], omegas["f2fast"] = f2fast(X, T)
        f["f"] = f["f1fast"] + f["f2fast"]
    else:
        raise ValueError("Unknown signal type")

    if noise:
        f["f"] += np.random.normal(0, noise_std, f["f"].shape)

    return f, omegas, x, t
