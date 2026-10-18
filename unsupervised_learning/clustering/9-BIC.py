#!/usr/bin/env python3
"""This module finds the best number of clusters using BIC for GMM"""

import numpy as np
expectation_maximization = __import__('8-EM').expectation_maximization


def BIC(X, kmin=1, kmax=None, iterations=1000, tol=1e-5, verbose=False):
    """Finds the best number of clusters using the BIC"""
    if not isinstance(X, np.ndarray) or X.ndim != 2:
        return None, None, None, None

    if not isinstance(kmin, int) or kmin < 1:
        return None, None, None, None

    if kmax is None:
        kmax = kmin

    if not isinstance(kmax, int) or kmax < kmin:
        return None, None, None, None

    if not isinstance(iterations, int) or iterations <= 0:
        return None, None, None, None

    if not isinstance(tol, float) or tol < 0:
        return None, None, None, None

    if not isinstance(verbose, bool):
        return None, None, None, None

    n, d = X.shape
    k_range = range(kmin, kmax + 1)
    num_k = len(k_range)
    log_likelihoods = np.empty(num_k)
    bics = np.empty(num_k)

    best_bic = np.inf
    best_k = None
    best_result = None

    for idx, k in enumerate(k_range):
        pi, m, S, g, ll = expectation_maximization(X, k, iterations, tol, verbose)
        if ll is None:
            return None, None, None, None

        # Parameters: k-1 priors, k means (d), k covariances (d * (d+1)/2)
        p = k * d + k * d * (d + 1) / 2 + k - 1
        bic = p * np.log(n) - 2 * ll

        log_likelihoods[idx] = ll
        bics[idx] = bic

        if bic < best_bic:
            best_bic = bic
            best_k = k
            best_result = (pi, m, S)

    return best_k, best_result, log_likelihoods, bics
