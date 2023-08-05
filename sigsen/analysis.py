"""
Module providing statistical modelling functionality
"""

import numpy as np
import sigsen.simulator as sim
from sigsen.maxima import maxima
from itertools import permutations
from dataclasses import dataclass


@dataclass
class SensorData:
    """
    A simple container class representing the sensor data from a single event

    Parameters
    ----------
    sensors: np.ndarray
        The locations of the sensors as a list of `(x,y)` pairs.
    sources: np.ndarray
        The locations of the sources as a list of `(x,y)` pairs.
    signals: np.ndarray
        The signal data ordered vertically by sensor and horizontally a list of
        signal arrival times.
    extent: tuple[float, ...]
        The bounding box of the sensor field in the format `(xmin, xmax, ymin, ymax)`.
    signal_speed: float
        The speed the signals travel at.
    noise: float
        The noise in the signals.
    log_distribution: np.ndarray | None, optional
        The prior (log-) distribution of signal sources (default is `None` which
        causes the algorithm to use a uniform prior).

    """
    sensors: np.ndarray
    sources: np.ndarray
    signals: np.ndarray

    extent: tuple[float, ...]
    signal_speed: float
    noise: float

    log_distribution: np.ndarray | None = None
    
    def _log_dist_error(self):
        """
        Checks whether the log distribution has been calculated yet

        ISSUE: this will return `True` if a prior is given, regardless of whether the
        data has actually been used to calculate the posterior.

        Returns
        -------
        bool
            Returns `True` if the log posterior distribution has been calculated.

        """
        if self.log_distribution is None:
            raise ValueError('The distribution has not yet been calculated.')

    def _idx_to_loc(
            self,
            i: int,
            j: int,
            resolution: tuple[int, int]
    ) -> np.ndarray:
        """
        Utility method to translate array index to cartesian coordinates.

        Parameters
        ----------
        i : int
            Row index in array
        j : int
            Column index in array
        resolution : tuple[int, int]
            The shape of the array, i.e. the size of the grid that the bounding box
            has been divided into.

        Returns
        -------
        np.ndarray
            A array of shape `(2,)` giving the cartesian coordinates of the centre of
            grid square at index `[i, j]`.

        """
        x_step = (self.extent[1] - self.extent[0]) / resolution[0]
        y_step = (self.extent[3] - self.extent[2]) / resolution[1]
        return np.array([
            self.extent[0] + (j + 0.5) * x_step,
            self.extent[3] - (i + 0.5) * y_step
        ])

    def signal_distribution(
            self,
            resolution: tuple[int, int],
            sensor_indices: tuple[int, ...] | None = None,
            source_indices: tuple[int, ...] | None = None,
    ) -> np.ndarray:
        """
        Calculates the log posterior distribution for the location of the sources.

        Parameters
        ----------
        resolution : tuple[int, int]
            The grid size (#rows, #columns), in which to break up the extent.
        sensor_indices : tuple[int, ...] | None, optional
            The indices of the sensors for which the data will be used (default is
            `None` which will use data from all sensors).
        source_indices : tuple[int, ...] | None, optional
            The indices of the sources for which the data will be used (default is
            `None` which will use data from all sources).

        Returns
        -------
        np.ndarray
            An array of shape `resolution` giving the log posterior distribution.

        """
        if sensor_indices is None:
            sensor_indices = tuple(range(len(self.sensors)))
        if source_indices is None:
            source_indices = tuple(range(len(self.sources)))

        log_dist = np.zeros(resolution[::-1])

        for sensor, signal in zip(
                self.sensors[sensor_indices, :],
                self.signals[np.ix_(sensor_indices, source_indices)]
        ):
            for idx, d in np.ndenumerate(log_dist):
                i, j = idx
                location = self._idx_to_loc(i, j, resolution)

                mean_shifted = signal - sim.dist(sensor, location)
                log_vals = -0.5 * np.square(mean_shifted / self.noise)
                log_dist[idx] += np.logaddexp.reduce(log_vals)

        log_dist -= np.logaddexp.reduce(log_dist, axis=None)
        return log_dist

    def compute_distribution(self, resolution: tuple[int, int]) -> None:
        """
        Compute the log posterior distribution for all data and save data.

        Parameters
        ----------
        resolution : tuple[int, int]
            The grid size (#rows, #columns), in which to break up the extent.

        Returns
        -------
        None

        """
        self.log_distribution = self.signal_distribution(resolution=resolution)

    def maxima_locs(self) -> np.ndarray:
        """
        Calculate the location of local maxima in the log posterior distribution.

        Returns
        -------
        np.ndarray
            The cartesian coordinates of the local maxima.

        """
        if self.log_distribution is None:
            raise ValueError('The log distribution has not yet been calculated')
        resolution = self.log_distribution.shape[::-1]
        maxima_idx = maxima(self.log_distribution, self.sources.shape[0])
        return np.array([self._idx_to_loc(i, j, resolution) for i, j in maxima_idx])

    def display(
            self,
            exp_factor: float = 0.01,
            show_sources: bool = True,
            show_maxima: bool = False,
    ):
        """
        Displays the posterior distribution.

        Parameters
        ----------
        exp_factor : float, optional
            A factor with which to scale the log posterior before taking the
            exponential. Useful to get sensible pictures (default is 0.1).
        show_sources : bool, optional
            If `True`, show the true location of the sources (default is `True`)
        show_maxima : bool, optional
            If `True`, show the calculated maxima in the posterior, i.e. the most
            likely locations of the sources according to the posterior (default is
            `True`).

        Returns
        -------
        matplotlib.Figure, matplotlib.Axes

        """
        range = sim.SensorField(*self.extent, sensor_locations=self.sensors)
        fig, ax = range.display()

        if self.log_distribution is not None:
            a = np.exp(exp_factor * self.log_distribution)
            ax.imshow(
                a,
                cmap='magma',
                interpolation='nearest',
                extent=self.extent,
                zorder=50,
                alpha=a/np.max(a),
            )

        if show_sources:
            ax.scatter(
                x=self.sources[:, 0],
                y=self.sources[:, 1],
                c='limegreen',
                marker='*',
                zorder=100,
                s=100,
            )
        if show_maxima:
            max_locs = self.maxima_locs()
            ax.scatter(
                x=max_locs[:, 0],
                y=max_locs[:, 1],
                c='red',
                marker='+',
                zorder=100,
            )

        return fig, ax

    def score(self) -> float:
        """
        The average distance between the true source location and the most likely
        location according to the posterior.

        Returns
        -------
        float
            The average score/error.

        """
        maxes = self.maxima_locs()
        srcs = self.sources
        perms = [list(p) for p in permutations(range(len(srcs)))]
        return float(
            min(np.sum(np.square(sim.dist(maxes, srcs[p]))) for p in perms)
        ) / len(srcs)
