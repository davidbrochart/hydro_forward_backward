import numpy as np
from scipy import stats

def get_kde(samples, a=np.inf, b=-np.inf):
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
    xy = np.empty((2, 100))
    xy[0] = np.linspace(smin, smax, 100)
    xy[1] = stats.gaussian_kde(samples)(xy[0])
    xy[1] = xy[1] / np.sum(xy[1])
    return xy

def uniform_density(a, b):
    xy = np.empty((2, 100))
    xy[0] = np.linspace(a, b, 100)
    xy[1, :] = 1
    return xy

def lnprob_from_density(xy):
    def lnprob(value):
        return np.log(np.interp(value, xy[0], xy[1], left=0, right=0))
        #if xy[0][0] <= value <= xy[0][-1]:
        #    v = np.interp(value, xy[0], xy[1])
        #else:
        #    # tail distribution, probability must not be 0 and must decrease
        #    vtail = xy[1][0]
        #    if value < xy[0][0]:
        #        e = value - xy[0][0]
        #    else:
        #        vtail = xy[1][-1]
        #        e = xy[0][-1] - value
        #    v = vtail * np.exp(e)
        #return np.log(v)
    return lnprob
