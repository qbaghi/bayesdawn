.. _imputation_methods:

Imputation methods
==================

BayesDawn allows you to choose the method among different algorithms to perform
the missing data imputation. This choice is important as it enables one to 
choose the trade-off between accuracy and compunational efficiency. Below we review the available methods.


Available methods
-----------------


* Nearest-neighboors ('nearest'): this methods use an approximation that estimates the missing data inside one gap conditionnaly on the observed data points directly located before and after the gap. This method can be parametrized by two parameters:

    - na : integer, number of observed data points to take into account before the gap
    - nb : integer, number of observed data points to take into account after the gap

* Preconditionned gradient algorithm ('PCG'): this method computes the exact conditional expectation of missing data given the entire observed data. It should be used when the number of missing data points is high (larger than 500), or when data gaps are long compared to the length of the time series. The inversion of the observed data covariance matrix is done iteratively using a conjugate gradient algorithm. The parameters are:

    - p : integer, number of data points to use when building the preconditionner, i.e., the matrix approximating the observed data covariance. Increasing p will increase convergence of the conjugate gradients, but will use a lot of RAM memory. We recommend using values smaller than 50.
    - tol : float, convergence criterium to meet : when the normalized residual error of the system solution falls below this value, the PCG aglorithm stops.
    - n_it_max : integer, maximum number of iteration of the PCG algorithm.

* Reduced-rank method ('woodbury'): this method computes the exact conditional expectation of missing data given the entire observed data, using a formula that is efficient when the number of missing data points is small (below than 500 or so). The same parameters as for the PCG method can be tuned for the offline precomputations.


Method use
----------

Here we describe how to specify which method to use. We assume that the quick_start_ section has been already run.
