"""
A module providing utilities for finding local maxima in numpy arrays
"""

import numpy as np
from itertools import product


def max_ij(a: np.ndarray) -> tuple[int, ...]:
    """
    Returns the index of the max entry of an array.

    Parameters
    ----------
    a : np.ndarray
        An array

    Returns
    -------
    tuple[int, ...]
        The index of the max entry of `a`.

    """
    return tuple(np.unravel_index(np.argmax(a), a.shape))


def loc_in_shape(
        loc: tuple[int, ...],
        shape: tuple[int, ...]
) -> bool:
    """
    Determines whether a given index is a valid location in an array of specified shape.

    Parameters
    ----------
    loc : tuple[int, ...]
        The index to be tested.
    shape : tuple[int, ...]
        The shape of the array.

    Returns
    -------
    bool
        Returns `True` if `loc` is a valid index for an array of shape `shape`'

    """
    if len(loc) != len(shape):
        raise ValueError(f'The index {loc} is not valid for shape {shape}')
    return all(0 <= x < a for x, a in zip(loc, shape))


def neighbours(
        loc: tuple[int, ...],
        shape: tuple[int, ...]
) -> list[tuple[int, ...]]:
    """
    Given an entry of an array, return a list of the neighbouring entries.

    Parameters
    ----------
    loc : tuple[int, ...]
        The index of the entry whose neighbours are to be found.
    shape : tuple[int, ...]
        The shape of the array.

    Returns
    -------
    list[tuple[int, ...]]
        A list of indices of the neighbouring entries.
    """
    dim = len(shape)
    if len(loc) != dim or not loc_in_shape(loc, shape):
        raise ValueError(f'The index {loc} is not valid for shape {shape}')
    maps = [zip(range(dim), i) for i in product((-1, 0, 1), repeat=dim)]
    n = [tuple(loc[i]+s for i, s in map) for map in maps]
    return [idx for idx in n if loc_in_shape(idx, shape)]


def lengthen(l: list, n: int) -> list:
    """
    Lengthen a list to specified length by repeating its entries.

    Parameters
    ----------
    l : list
        The list to be lengthened.
    n : int
        The total length desired.

    Returns
    -------
    list
        The lengthened list.

    """
    idx = 0
    long_l = []
    while len(long_l) < n:
        long_l += [l[idx]]
        idx = (idx + 1) % len(l)
    return long_l


def maxima(a: np.ndarray, n: int) -> list[tuple[int, ...], ...]:
    """
    Returns a list of the local maxima of an array.

    Parameters
    ----------
    a : np.ndarray
        The array whose local maxima are to be found.
    n : int
        The number of local maxima desired.

    Returns
    -------
    list[tuple[int, ...], ...]
        A list of indices to the local maxima.

    """
    x = np.copy(a)
    max_locations = []
    while len(max_locations) < n:
        max_loc = max_ij(x)
        if x[max_loc] == -np.Inf:
            max_locations = lengthen(max_locations, n)
            break
        if -np.Inf not in [x[n] for n in neighbours(max_loc, x.shape)]:
            max_locations += [max_loc]
        x[max_loc] = -np.Inf
    return max_locations
