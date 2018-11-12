import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from pandas import DataFrame

def get_kde(samples, a=np.inf, b=-np.inf, nb=100):
    samples = samples[np.isfinite(samples)]
    smin, smax = np.min(samples), np.max(samples)
    if smin == smax:
        smin *= 0.99
        smax *= 1.01
        samples[:2] = [smin, smax]
    if a < smin:
        smin = a
    if b > smax:
        smax = b
    xy = np.empty((2, nb))
    xy[0] = np.linspace(smin, smax, nb)
    xy[1] = stats.gaussian_kde(samples)(xy[0])
    xy[1] /= np.trapz(xy[1], x=xy[0])
    return xy

def uniform_density(a, b):
    xy = np.empty((2, 100))
    xy[0] = np.linspace(a, b, 100)
    xy[1, :] = 1
    xy[1] /= np.trapz(xy[1], x=xy[0])
    return xy

def lnprob_from_density(xy, vmin=None, vmax=None):
    def lnprob(value):
        # the simple way, everything outside the possible values has 0 probability
        # return np.log(np.interp(value, xy[0], xy[1], left=0, right=0))
        if ((vmin is not None) and (value < vmin)) or ((vmax is not None) and (value > vmax)):
            return -np.inf
        if xy[0][0] <= value <= xy[0][-1]:
            v = np.interp(value, xy[0], xy[1])
        else:
            # tail distribution, probability must not be 0 and must decrease
            if value < xy[0][0]:
                vtail = xy[1][0]
                e = value - xy[0][0] # negative
            else:
                vtail = xy[1][-1]
                e = xy[0][-1] - value # negative
            e /= xy[0][-1] - xy[0][0] # normalization
            v = vtail * np.exp(e)
        return np.log(v)
    return lnprob

def plot_series(ensemble=None, true=None, mean=None, measure=None, title=''):
    plt.figure(figsize=(17, 5))
    if ensemble is not None:
        n = 1000 # number of time series from the ensemble to plot
        alpha = 10 / n
        if len(ensemble) > n:
            step = len(ensemble) // n
        else:
            step = 1
        for i in range(0, len(ensemble), step):
            plt.plot(ensemble[i], color='red', alpha=alpha)
        plt.plot([0], [0], color='red', alpha=0.5, label='Simulated')
    if true is not None:
        plt.plot(true, color='green', linestyle='dashed', alpha=0.5, label='True')
    if measure is not None:
        plt.scatter(np.arange(true.size), measure, label='Measured', color='green', alpha=0.5)
    if mean is not None:
        plt.plot(mean, color='blue', label='Mean')
    plt.title(title)
    plt.legend()
    plt.show()

def dist_map(x, y):
    df = DataFrame({'x': x, 'y': y}).dropna()
    x_sorted = np.sort(df.x.values)
    y_sorted = np.sort(df.y.values)
    return np.interp(x, x_sorted, y_sorted)
