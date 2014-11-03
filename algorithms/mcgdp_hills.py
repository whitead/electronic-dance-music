import numpy as np
import matplotlib.pyplot as plt
from math import *
from scipy.special import erf


def make_hill(x, min, max, sigma):
    xx = np.arange(min,max,0.01)
    
    C =  0.5
    denom = C * (erf((xx - min) / (sqrt(2) * sigma)) + erf((max - xx) / (sqrt(2) * sigma)))
    expo = 1.5 * np.exp(-(xx - x)**2 / (2 * sigma**2))
    vol1 = np.sum(expo / denom * 0.01)
    vol2 = np.sum(expo * 0.01)
    return xx, expo / denom, vol1, vol2

x,y,v1,v2 = make_hill(0, -100, 100, 2)

print v1,v2

plt.plot(x,y)
plt.savefig("hill.png")
