import numpy as np
import matplotlib.pyplot as plt
from math import *
from scipy.special import erf

def sigmoid(x):
    if(x > 1):
        return 0
    if(x < 0):
        return 1
    return 2. * x**3 - 3. * x ** 2 + 1

v_sigmoid = np.vectorize(sigmoid)


def make_hill(x, min, max, sigma):
    xx = np.arange(min,max,0.01)  
    expo = np.exp(-(xx - x)**2 / (2*sigma**2))
    denom = sqrt(2*pi) * sigma / 2. * (erf((xx - min) / (sqrt(2.) * sigma)) + erf((max - xx) / (sqrt(2.) * sigma)))
    func_denom = denom + (0.5*sqrt(2*pi)*sigma - denom) * v_sigmoid((xx - min)/(sqrt(2.)*sigma))  + (0.5*sqrt(2*pi)*sigma - denom) * v_sigmoid((max - xx)/(sqrt(2.)*sigma)) 
    func = expo +  (exp(-(x-min)**2 / (2*sigma**2)) - expo) * v_sigmoid((xx-min)/(sqrt(2.)*sigma)) + (exp(-(x-max)**2 / (2.*sigma**2)) - expo) * v_sigmoid((max-xx)/(sqrt(2.)*sigma))
    
    return xx, func / func_denom, expo / denom

x,y1,y2 = make_hill(0., 0., 10., 2.)
plt.plot(x,y1, color="blue")
plt.plot(x,y2, color="blue", linestyle='--')

x,y1,y2 = make_hill(1., 0., 10., 2.)
plt.plot(x,y1, color="green")
plt.plot(x,y2, color="green", linestyle='--')

x,y1,y2 = make_hill(2., 0., 10., 2.)
plt.plot(x,y1, color="red")
plt.plot(x,y2, color="red", linestyle='--')

#x,y = make_hill(9.8, 0, 10, 2)
#plt.plot(x,y)

plt.savefig("hill.png")
