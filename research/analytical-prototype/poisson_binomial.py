#!/usr/bin/env python
# -*-coding:utf-8 -*-
"""Exact distribution of a weighted sum of independent Bernoullis.

A confidence-pool score is  S = sum_i  w_i * X_i,  where X_i ~ Bernoulli(p_i)
is "pick i was correct" and w_i is the integer confidence assigned to game i.
With integer weights S takes integer values in [0, sum(w_i)] and its PMF is a
convolution of the per-game two-point distributions -- no simulation needed.

This module is written to be promotion-ready (no repo deps); the prototype
notebook imports it directly.
"""
from __future__ import annotations

from typing import Iterable, Sequence, Tuple

import numpy as np


def weighted_pmf(probs: Sequence[float], weights: Sequence[int]) -> np.ndarray:
    """PMF of S = sum_i weights[i] * Bernoulli(probs[i]).

    Args:
        probs:   per-game success probabilities, each in [0, 1].
        weights: per-game integer weights (confidence points), each >= 0.

    Returns:
        1-D array ``pmf`` of length ``sum(weights) + 1`` where ``pmf[k]`` is
        P(S == k). Sums to 1 (within floating point).
    """
    probs = np.asarray(probs, dtype=float)
    weights = np.asarray(weights, dtype=int)
    if probs.shape != weights.shape:
        raise ValueError("probs and weights must have the same length")
    if np.any((probs < 0) | (probs > 1)):
        raise ValueError("probs must be in [0, 1]")
    if np.any(weights < 0):
        raise ValueError("weights must be non-negative integers")

    total = int(weights.sum())
    pmf = np.zeros(total + 1)
    pmf[0] = 1.0
    size = 1  # current support is pmf[:size]

    for p, w in zip(probs, weights):
        if w == 0:
            continue  # a zero-weight game never moves the score
        new_size = size + w
        shifted = np.zeros(new_size)
        shifted[:size] += pmf[:size] * (1.0 - p)   # game i wrong
        shifted[w:w + size] += pmf[:size] * p      # game i right, +w
        pmf[:new_size] = shifted
        size = new_size

    return pmf[:size]


def pmf_via_fft(probs: Sequence[float], weights: Sequence[int]) -> np.ndarray:
    """Same result as :func:`weighted_pmf` via FFT convolution.

    Faster when there are many games and the total is large; kept mainly to
    cross-check the direct version. Small negative values from round-off are
    clipped to 0 and the result is renormalized.
    """
    probs = np.asarray(probs, dtype=float)
    weights = np.asarray(weights, dtype=int)
    total = int(weights.sum())
    n = total + 1

    # Each game contributes a length-(w+1) kernel [1-p, 0, ..., 0, p]; in the
    # DFT domain that is (1-p) + p * omega**w with omega = exp(-2 pi i / n).
    spectrum = np.ones(n, dtype=complex)
    omega = np.exp(-2j * np.pi * np.arange(n) / n)
    for p, w in zip(probs, weights):
        if w == 0:
            continue
        spectrum *= (1.0 - p) + p * omega ** w

    pmf = np.fft.ifft(spectrum).real
    pmf = np.clip(pmf, 0.0, None)
    s = pmf.sum()
    return pmf / s if s > 0 else pmf


def survival(pmf: np.ndarray) -> np.ndarray:
    """P(S >= k) for k = 0 .. len(pmf)-1, given a PMF over 0 .. len(pmf)-1."""
    return np.cumsum(pmf[::-1])[::-1]


def prob_a_beats_b(pmf_a: np.ndarray, pmf_b: np.ndarray,
                   tie_credit: float = 0.5) -> float:
    """P(A > B) + tie_credit * P(A == B) for independent scores A, B.

    A, B are PMFs over 0..len-1. Uses  P(A > B) = sum_k P(A = k) * P(B <= k-1).
    """
    n = max(len(pmf_a), len(pmf_b))
    a = np.zeros(n)
    a[:len(pmf_a)] = pmf_a
    b = np.zeros(n)
    b[:len(pmf_b)] = pmf_b
    cdf_b = np.cumsum(b)                       # cdf_b[k] = P(B <= k)
    # P(B <= k-1): shift right by one, P(B <= -1) = 0
    cdf_b_lt = np.concatenate(([0.0], cdf_b[:-1]))
    p_strict = float(np.dot(a, cdf_b_lt))
    p_tie = float(np.dot(a, b))
    return p_strict + tie_credit * p_tie


def _brute_force_pmf(probs: Sequence[float], weights: Sequence[int]) -> np.ndarray:
    """Reference PMF by enumerating all 2**n outcomes. Only for tests / small n."""
    probs = list(probs)
    weights = list(weights)
    n = len(probs)
    total = int(sum(weights))
    pmf = np.zeros(total + 1)
    for mask in range(1 << n):
        p = 1.0
        s = 0
        for i in range(n):
            if mask & (1 << i):
                p *= probs[i]
                s += weights[i]
            else:
                p *= 1.0 - probs[i]
        pmf[s] += p
    return pmf


def _self_check() -> None:
    rng = np.random.default_rng(0)
    for _ in range(200):
        n = int(rng.integers(1, 9))
        probs = rng.random(n)
        weights = rng.integers(0, 6, size=n)
        ref = _brute_force_pmf(probs, weights)
        for name, pmf in (("direct", weighted_pmf(probs, weights)),
                          ("fft", pmf_via_fft(probs, weights))):
            if len(pmf) != len(ref):
                raise AssertionError(f"{name}: length {len(pmf)} != {len(ref)}")
            if not np.allclose(pmf, ref, atol=1e-9):
                raise AssertionError(f"{name}: max abs err {np.abs(pmf - ref).max():.2e}")

    # prob_a_beats_b against brute force
    for _ in range(100):
        na, nb = int(rng.integers(1, 6)), int(rng.integers(1, 6))
        pa, wa = rng.random(na), rng.integers(1, 5, size=na)
        pb, wb = rng.random(nb), rng.integers(1, 5, size=nb)
        A = weighted_pmf(pa, wa)
        B = weighted_pmf(pb, wb)
        got = prob_a_beats_b(A, B)
        # brute: P(A>B) + 0.5 P(A==B)
        ref = 0.0
        for ka, va in enumerate(A):
            for kb, vb in enumerate(B):
                if ka > kb:
                    ref += va * vb
                elif ka == kb:
                    ref += 0.5 * va * vb
        if abs(got - ref) > 1e-9:
            raise AssertionError(f"prob_a_beats_b err {abs(got - ref):.2e}")

    print("poisson_binomial self-check passed")


if __name__ == "__main__":
    _self_check()
