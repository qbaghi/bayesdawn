# Import mecm and other useful packages
from bayesdawn import imputation
import numpy as np
import random
from scipy import signal


if __name__=='__main__':

    # Choose size of data
    N = 2 ** 14
    # Generate Gaussian white noise
    noise = np.random.normal(loc=0.0, scale=1.0, size=N)
    # Apply filtering to turn it into colored noise
    r = 0.01
    b, a = signal.butter(3, 0.1 / 0.5, btype='high', analog=False)
    n = signal.lfilter(b, a, noise, axis=-1, zi=None) + noise * r



