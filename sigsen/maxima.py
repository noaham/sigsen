"""
A module providing utilities for finding local maxima in numpy arrays
"""

import numpy as np


def max_ij(a: np.ndarray) -> tuple[int, ...]:
    return tuple(np.unravel_index(np.argmax(a), a.shape))


def neighbours(loc: tuple[int, ...], a: np.ndarray) -> list[tuple[int, int]]:
    n = []
    if loc[0] > 0:
        n += [(loc[0] - 1, loc[1])]
    if loc[1] > 0:
        n += [(loc[0], loc[1] - 1)]
    if loc[0] < a.shape[0] - 1:
        n += [(loc[0] + 1, loc[1])]
    if loc[1] < a.shape[1] - 1:
        n += [(loc[0], loc[1] + 1)]
    return n


def maxima(a: np.ndarray, n: int) -> list[tuple[int, ...], ...]:
    x = np.copy(a)
    max_locations = []
    while len(max_locations) < n:
        max_loc = max_ij(x)
        if -np.Inf not in [x[n] for n in neighbours(max_loc, x)]:
            max_locations += [max_loc]
        x[max_loc] = -np.Inf
    return max_locations


