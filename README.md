BayesDawn
=================



BAYESDAWN stands for Bayesian Data Augmentation for Waves and Noise. It implements an iterative Bayesian augmentation 
method to handle data gaps in gravitational-wave data analysis, as described in this paper: https://arxiv.org/abs/1907.04747.

Installation
------------

BayesDawn can be installed by unzipping the source code in one directory and using this command: ::

    sudo python setup.py install
    
    
Quick start
-----------

Using BayesDawn for your own analysis will essentially involve the imputation.py module, which allows you to 
compute the conditional distribution of missing values given the observed values of a time series.
Here is a working example that can be used.

1. Generation of test data

To begin with, we generate some simple time series which contains noise and signal.
To generate the noise, we start with a white, zero-mean Gaussian noise that
we then filter to obtain a stationary colored noise:

.. code-block::

  # Import mecm and other useful packages
  from bayesdawn import imputation
  import numpy as np
  import random
  from scipy import signal
  # Choose size of data
  N = 2**14
  # Generate Gaussian white noise
  noise = np.random.normal(loc=0.0, scale=1.0, size = N)
  # Apply filtering to turn it into colored noise
  r = 0.01
  b, a = signal.butter(3, 0.1/0.5, btype='high', analog=False)
  n = signal.lfilter(b,a, noise, axis=-1, zi=None) + noise*r

Then we need a deterministic signal to add. We choose a sinusoid with some
frequency f0 and amplitude a0:

.. code-block::

  t = np.arange(0,N)
  f0 = 1e-2
  a0 = 5e-3
  s = a0*np.sin(2*np.pi*f0*t)

We just have generated a time series that can be written in the form

.. math::

  y = A \beta + n

Now assume that some data are missing, i.e. the time series is cut by random gaps.
The pattern is represented by a mask vector M with entries equal to 1 when data
is observed, and 0 otherwise:

.. code-block::

  mask = np.ones(N)
  Ngaps = 30
  gapstarts = (N*np.random.random(Ngaps)).astype(int)
  gaplength = 10
  gapends = (gapstarts+gaplength).astype(int)
  for k in range(Ngaps): mask[gapstarts[k]:gapends[k]]= 0

Therefore, we do not observe y but its masked version, mask*y.