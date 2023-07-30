"""
A module providing utilities for finding local maxima in numpy arrays
"""

import numpy as np
from itertools import product


def max_ij(a: np.ndarray) -> tuple[int, ...]:
    return tuple(np.unravel_index(np.argmax(a), a.shape))


def loc_in_shape(
        loc: tuple[int, ...],
        shape: tuple[int, ...]
) -> bool:
    if len(loc) != len(shape):
        raise ValueError(f'The index {loc} is not valid for shape {shape}')
    return all(0 <= x < a for x, a in zip(loc, shape))


def neighbours(
        loc: tuple[int, ...],
        shape: tuple[int, ...]
) -> list[tuple[int, ...]]:
    dim = len(shape)
    if len(loc) != dim or not loc_in_shape(loc, shape):
        raise ValueError(f'The index {loc} is not valid for shape {shape}')
    maps = [zip(range(dim), i) for i in product((-1, 0, 1), repeat=dim)]
    n = [tuple(loc[i]+s for i, s in map) for map in maps]
    return [idx for idx in n if loc_in_shape(idx, shape)]


def maxima(a: np.ndarray, n: int) -> list[tuple[int, ...], ...]:
    x = np.copy(a)
    max_locations = []
    while len(max_locations) < n:
        max_loc = max_ij(x)
        if -np.Inf not in [x[n] for n in neighbours(max_loc, x.shape)]:
            max_locations += [max_loc]
        x[max_loc] = -np.Inf
    return max_locations
