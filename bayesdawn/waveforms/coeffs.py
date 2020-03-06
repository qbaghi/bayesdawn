#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb  1 15:36:02 2019

@author: qbaghi
"""

import numpy as np


def k_coeffs(params, Phi_rot, i, j):
    """

    This function calculates k^{ij}_{+,m} and k^{ij}_{x,m} where

    k^{ij}_{alpha,m} = a^{ij}_{alpha,m} - j b^{ij}_{alpha,m}

    and a^{ij}_{alpha,m} = a^{i}_{alpha,m} - a^{j}_{alpha,m}
    b^{ij}_{alpha,m} = b^{i}_{alpha,m} - b^{j}_{alpha,m}

    coeffs_plus = np.concatenate((da_p,db_p))
    coeffs_cros = np.concatenate((da_c,db_c))

    Parameters
    ----------
    params : array_like
        vector of extrinsinc parameters: theta,phi,f_0,f_dot
    Phi_rot : scalar float
        initial angle of LISA constellation
    i : scalar integer
        index of first arm
    j : scalar integer
        index of second arm

    """

    theta = params[0]
    phi = params[1]

    # Two vectors of size 10: coeffs_p = da_p,db_p and coeffs_c = da_c,db_c
    coeffs_p, coeffs_c = xi_diff_coeffs(theta, phi, Phi_rot, i, j)

    k_p = coeffs_p[0:5] - 1j*coeffs_p[5:10]
    k_c = coeffs_c[0:5] - 1j*coeffs_c[5:10]

    return k_p, k_c


def xi_diff_coeffs(theta, phi, Phi_rot, i, j):
    """
    Create the python function to compute the coefficient vector ai_diff in

    xi_diff = A_cs *ai_diff

    where xi_diff is the difference of the projection
    functionals xi_+(t) - xj_+(t)
    such that the strain on the barycentric frame projects as
    H(t,tau) = h_{B+}(tau) xi_+(t) + h_{Bx}(tau) xi_x(t)

    Parameters
    ----------
    theta : scalar float
        colatitude angle: theta = beta + pi/2 if beta is the ecliptic latitude
    phi : scalar float
        longitude angle such taht phi = lam - pi where lam is the ecliptic
        longitude
    Phi_rot : scalar float
        rotation angle between the initial configuration of the LISA triangle
        and the standard initial configuration (reference) which is so that
        at t=0 the triangle edge where S/C 1 is located poits downwards (wrt
        ecliptic plane), S/C 2 is on the y<0 side and S/C 3 is on the third
        edge.
    i : integer float {1,2,3}
        index of first spacecraft
    j : integer float {1,2,3}
        index of second spacecraft

    Returns
    -------
    coeffs_plus : numpy 1d array
        modulation coefficients of the + polarization
    coeffs_cros : numpy 1d array
        modulation coefficients of the x polarization


    """

    Phi_plus = (i + j + 1) * 4 * np.pi / 3 - 2 * Phi_rot
    Phi_minus = 2 * (i - j) * np.pi / 3

    costheta = np.cos(theta)
    sintheta = np.sin(theta)
    cos2theta = np.cos(2 * theta)
    sin2theta = np.sin(2 * theta)
    sinPhi_minus = np.sin(Phi_minus)

    # Compute the coefficients for + polarization
    da_p = np.zeros(5, dtype=np.float64)
    da_p[0] = -9 / 32 * np.sin(2 * phi - Phi_plus) * (cos2theta + 3)
    da_p[1] = -3 * np.sqrt(3) / 8 * sin2theta * np.sin(-phi + Phi_plus)
    da_p[2] = 9 / 16 * (1 - cos2theta) * np.sin(Phi_plus)
    da_p[3] = np.sqrt(3) / 8 * sin2theta * np.sin(phi + Phi_plus)
    da_p[4] = 1 / 32. * (cos2theta + 3) * np.sin(2 * phi + Phi_plus)
    da_p = sinPhi_minus * da_p

    db_p = np.zeros(5, dtype=np.float64)
    db_p[0] = 0
    db_p[1] = 3 * np.sqrt(3) / 8 * sin2theta * np.cos(-phi + Phi_plus)
    db_p[2] = -9 / 16 * (1 - cos2theta) * np.cos(Phi_plus)
    db_p[3] = -np.sqrt(3) / 8 * sin2theta * np.cos(phi + Phi_plus)
    db_p[4] = -1 / 32. * (cos2theta + 3) * np.cos(2 * phi + Phi_plus)

    db_p = sinPhi_minus * db_p

    # Compute coefficients for x polarization
    da_c = np.zeros(5, dtype=np.float64)
    da_c[0] = - 9 / 8. * costheta * np.cos(2 * phi - Phi_plus)
    da_c[1] = 3 / 4. * np.sqrt(3) * sintheta * np.cos(phi - Phi_plus)
    da_c[2] = 0
    da_c[3] = (1 / 4.) * np.sqrt(3) * sintheta * np.cos(phi + Phi_plus)
    da_c[4] = 1 / 8. * costheta * np.cos(2 * phi + Phi_plus)

    da_c = sinPhi_minus * da_c

    db_c = np.zeros(5, dtype=np.float64)
    db_c[0] = 0
    db_c[1] = - 3 / 4. * np.sqrt(3) * sintheta * np.sin(phi - Phi_plus)
    db_c[2] = 0
    db_c[3] = 1 / 4. * np.sqrt(3) * sintheta * np.sin(phi + Phi_plus)
    db_c[4] = 1 / 8. * costheta * np.sin(2 * phi + Phi_plus)

    db_c = sinPhi_minus * db_c

    coeffs_plus = np.concatenate((da_p, db_p))
    coeffs_cros = np.concatenate((da_c, db_c))

    return coeffs_plus, coeffs_cros


def k_coeffs_single(params, phi_rot, i):
    """

    This function calculates k^{i}_{+,m} and k^{i}_{x,m} where

    k^{i}_{alpha,m} = a^{i}_{alpha,m} - j b^{i}_{alpha,m}

    coeffs_plus = np.concatenate((a_p, b_p))
    coeffs_cros = np.concatenate((a_c, b_c))

    Parameters
    ----------
    params : array_like
        vector of extrinsinc parameters: theta,phi,f_0,f_dot
    phi_rot : scalar float
        initial angle of LISA constellation
    i : scalar integer
        index of single-link (arm)

    """

    theta = params[0]
    phi = params[1]

    # Two vectors of size 10: coeffs_p = da_p,db_p and coeffs_c = da_c,db_c
    coeffs_p, coeffs_c = xi_coeffs(theta, phi, phi_rot, i)

    k_p = coeffs_p[0:5] - 1j*coeffs_p[5:10]
    k_c = coeffs_c[0:5] - 1j*coeffs_c[5:10]

    return k_p, k_c


def xi_coeffs(theta, phi, phi_rot, i):
    """
    Create the python function to compute the coefficient vector ai_diff in

    xi = A_cs * ai

    where xi is the projection functional xi_+(t) such that the strain on the
    barycentric frame projects as
    H(t,tau) = h_{B+}(tau) xi_+(t) + h_{Bx}(tau) xi_x(t)

    Parameters
    ----------
    theta : scalar float
        colatitude angle: theta = beta + pi/2 if beta is the ecliptic latitude
    phi : scalar float
        longitude angle such taht phi = lam - pi where lam is the ecliptic
        longitude
    phi_rot : scalar float
        rotation angle between the initial configuration of the LISA triangle
        and the standard initial configuration (reference) which is so that
        at t=0 the triangle edge where S/C 1 is located poits downwards (wrt
        ecliptic plane), S/C 2 is on the y<0 side and S/C 3 is on the third
        edge.
    i : integer float {1,2,3}
        index of spacecraft (or single-link channel)

    Returns
    -------
    coeffs_plus : numpy 1d array
        modulation coefficients of the + polarization
    coeffs_cros : numpy 1d array
        modulation coefficients of the x polarization


    """

    # THIS IS NOT TRUE: THIS IS PHI_ROT
    # phi_rot_i = 2 * (i-1) * np.pi/3 - phi_rot
    # Should be Phi_i = 0.5 * (phi_rot_{i+2} + phi_rot_{i+1})
    phi_i = np.pi / 3 * (2 * i + 1) - phi_rot

    costheta = np.cos(theta)
    sintheta = np.sin(theta)
    cos2theta = np.cos(2 * theta)
    sin2theta = np.sin(2 * theta)

    # Compute the coefficients for + polarization
    a_p = np.zeros(5, dtype=np.float64)
    a_p[0] = -1/64 * (9 * np.cos(2 * phi - 2 * phi_i) * (cos2theta + 3) + 2 * (cos2theta - 1))
    a_p[1] = np.sqrt(3) / 16 * sin2theta * (3 * np.cos(phi - 2 * phi_i) - 2 * np.cos(phi))
    # a_p[2] = 3 / 32 * (3 * np.cos(2 * phi_i) * (cos2theta - 2) - np.cos(2 * phi) * (cos2theta + 6))
    a_p[2] = 3 / 32 * (3 * np.cos(2 * phi_i) * (cos2theta - 1) - np.cos(2 * phi) * (cos2theta + 3))
    a_p[3] = - np.sqrt(3) / 16 * sin2theta * np.cos(phi + 2 * phi_i)
    a_p[4] = - 1/64 * np.cos(2 * phi + 2 * phi_i) * (cos2theta + 3)

    b_p = np.zeros(5, dtype=np.float64)
    b_p[0] = 0
    b_p[1] = - np.sqrt(3) / 16 * sin2theta * (3 * np.sin(phi - 2 * phi_i) + 2 * np.sin(phi))
    # b_p[2] = 3 / 32 * (3 * np.sin(2 * phi_i) * (cos2theta - 2) - np.sin(2 * phi) * (cos2theta + 6))
    b_p[2] = 3 / 32 * (3 * np.sin(2 * phi_i) * (cos2theta - 1) - np.sin(2 * phi) * (cos2theta + 3))
    b_p[3] = - np.sqrt(3) / 16 * sin2theta * np.sin(phi + 2 * phi_i)
    b_p[4] = - 1/64 * np.sin(2 * phi + 2 * phi_i) * (cos2theta + 3)

    # Compute coefficients for x polarization
    a_c = np.zeros(5, dtype=np.float64)
    a_c[0] = 9/16 * costheta * np.sin(2 * phi - 2 * phi_i)
    a_c[1] = np.sqrt(3) / 8 * sintheta * (2 * np.sin(phi) - 3 * np.sin(phi - 2 * phi_i))
    a_c[2] = 3 / 8 * costheta * np.sin(2 * phi)
    a_c[3] = np.sqrt(3) / 8 * sintheta * np.sin(phi + 2 * phi_i)
    a_c[4] = 1 / 16 * costheta * np.sin(2 * phi + 2 * phi_i)

    b_c = np.zeros(5, dtype=np.float64)
    b_c[0] = 0
    b_c[1] = - np.sqrt(3) / 8 * sintheta * (2 * np.cos(phi) + 3 * np.cos(phi - 2 * phi_i))
    b_c[2] = - 3 / 8 * costheta * np.cos(2 * phi)
    b_c[3] = - np.sqrt(3) / 8 * sintheta * np.cos(phi + 2 * phi_i)
    b_c[4] = - 1 / 16 * costheta * np.cos(2 * phi + 2 * phi_i)

    coeffs_plus = np.concatenate((a_p, b_p))
    coeffs_cros = np.concatenate((a_c, b_c))

    return coeffs_plus, coeffs_cros


def beta_gb(a0, incl, phi_0, psi):
    """
    Create symbolic mathematical expression giving the vector beta of parameters
    in the matrix decomposition of the TDI response:

    TDI(t) = A(theta,phi,f0,f_dot,t) beta(A_p,A_c,phi_0,psi)

    Parameters
    ----------
    a0 : scalar float
        amplitude of the gravitational wave
    incl : scalar integer
        inclination of the binary orbit
    phi_0 : scalar float
        initial phase of the binary orbit
    psi : scalar float
        gravitational wave polarization angle


    Returns
    -------

    beta : sympy symbolic matrix
        vector of source parameters in the linear decomposition of the TDI
    """

    A_p = a0 * (1 + np.cos(incl) ** 2)
    A_c = -2 * a0 * np.cos(incl)

    C_p = A_p * np.cos(phi_0)
    S_p = - A_p * np.sin(phi_0)
    C_c = A_c * np.sin(phi_0)
    S_c = A_c * np.cos(phi_0)

    beta = np.array(
        [C_p * np.cos(2 * psi) + C_c * np.sin(2 * psi),
         - C_p * np.sin(2 * psi) + C_c * np.cos(2 * psi),
         S_p * np.cos(2 * psi) + S_c * np.sin(2 * psi),
         - S_p * np.sin(2 * psi) + S_c * np.cos(2 * psi)])

    return beta


def beta_gb_complex(a0, incl, phi_0, psi):
    """
    For the complex amplitude decomposition

    """

    beta = beta_gb(a0, incl, phi_0, psi)

    beta_p = beta[0] - 1j*beta[2]
    beta_c = beta[1] - 1j*beta[3]

    return np.array([beta_p, beta_c])
