BayesDawn
=================



BayesDawn stands for Bayesian Data Augmentation for Waves and Noise. It implements an iterative Bayesian augmentation 
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

```python
  # Import mecm and other useful packages
  from bayesdawn import imputation
  from bayesdawn.psd import psdspline
  import numpy as np
  import random
  from scipy import signal
  # Choose size of data
  N = 2**14
  # Set sampling frequency
  fs = 1.0
  # Generate Gaussian white noise
  noise = np.random.normal(loc=0.0, scale=1.0, size = N)
  # Apply filtering to turn it into colored noise
  r = 0.01
  b, a = signal.butter(3, 0.1/0.5, btype='high', analog=False)
  n = signal.lfilter(b,a, noise, axis=-1, zi=None) + noise*r
```

Then we need a deterministic signal to add. We choose a sinusoid with some
frequency f0 and amplitude a0:

```python
  t = np.arange(0,N) / fs
  f0 = 1e-2
  a0 = 5e-3
  s = a0*np.sin(2*np.pi*f0*t)
```

The noisy data is then

```python
  y = s + n
```

2. Introduction of data gaps

Now assume that some data are missing, i.e. the time series is cut by random gaps.
The pattern is represented by a mask vector with entries equal to 1 when data
is observed, and 0 otherwise:

```python
  mask = np.ones(N)
  Ngaps = 30
  gapstarts = (N * np.random.random(Ngaps)).astype(int)
  gaplength = 10
  gapends = (gapstarts+gaplength).astype(int)
  for k in range(Ngaps): mask[gapstarts[k]:gapends[k]]= 0
```

Therefore, we do not observe the data y but its masked version, mask*y:

```python
  y = s + n
  y = mask * y
```

3. Missing data imputation

Assune that we know exactly the deterministic signal:

```python
   s_model = s[:]
```
Then we can do a crude estimation of the PSD from masked data:

```python
    # Fit PSD with a spline of degree 2 and 10 knots
    psd_cls = psdspline.PSDSpline(N, fs, J=10, D=2, fmin=fs/N, fmax=fs/2)
    psd_cls.estimate(y - s_mask)
    psd = psd_cls.calculate(N)

```

Then, from the observed data and their model, we can reconstruct the missing data using the imputation package:

```python

    # instantiate imputation class
    imp_cls = imputation.nearestNeighboor(mask, Na=50, Nb=50)
    # Imputation of missing data
    y_rec = imp_cls.draw_missing_data(y_mask, s, psd_cls)


```


