"""
Module providing statistical modelling functionality
"""

import numpy as np
import sigsen.simulator as sim
from sigsen.maxima import maxima
from itertools import permutations
from dataclasses import dataclass
import matplotlib.animation as animation
import matplotlib.pyplot as plt
from matplotlib import colormaps
from matplotlib.colors import LinearSegmentedColormap as LSCmap, Normalize, Colormap

plt.style.use('seaborn-v0_8')


def cmap_with_alpha(cmap_name: str) -> Colormap:
    """
    Takes a standard matplotlib colormap and returns the same colormap with alphas
    ranging from 0 to 1 linearly.

    Parameters
    ----------
    cmap_name : str
        Name of the colormap.

    Returns
    -------
    matplotlib.colors.Colormap
        The transformed colormap.

    """
    cmap = colormaps[cmap_name]
    alphas = np.arange(256) / 256
    colors = cmap(alphas)
    colors[:, -1] = alphas
    return LSCmap.from_list("magma_alpha", colors)


MAGMA_ALPHAS_cmap = cmap_with_alpha("magma")


def permute21(a: tuple) -> tuple:
    """
    Utility function to permute the first two entries of a tuple.

    Parameters
    ----------
    a : tuple
        The tuple

    Returns
    -------
    tuple
        The original tuple with the first two entries swapped.

    """
    return (a[1], a[0]) + a[2:]


@dataclass
class SensorData:
    """
    A simple container class representing the sensor data from a single event

    Parameters
    ----------
    sensors: np.ndarray
        The locations of the sensors as a list of `(x,y)` pairs.
    sources: np.ndarray
        The locations of the sources as a list of `(x,y,t)` triples.
    signals: np.ndarray
        The signal data ordered vertically by sensor and horizontally a list of
        signal arrival times.
    extent: tuple[float, ...]
        The bounding box of the sensor field in the format `(xmin, xmax, ymin, ymax)`.
    t_max: float
        The time period in which events occur.
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
    t_max: float
    signal_speed: float
    noise: float

    log_distribution: np.ndarray | None = None
    _dist_calculated: bool = False
    
    def _log_dist_error(self):
        """
        Checks whether the log distribution has been calculated yet

        Returns
        -------
        bool
            Returns `True` if the log posterior distribution has been calculated.

        """
        if not self._dist_calculated:
            raise ValueError('The distribution has not yet been calculated.')

    def _idx_to_loc(
            self,
            i: int,
            j: int,
            k: int,
            resolution: tuple[int, int, int]
    ) -> np.ndarray:
        """
        Utility method to translate array index to cartesian coordinates.

        Parameters
        ----------
        i : int
            Row index in array
        j : int
            Column index in array
        k : int
            Sheet index in array
        resolution : tuple[int, int, int]
            The shape of the array, i.e. the size of the grid that the bounding box
            has been divided into.

        Returns
        -------
        np.ndarray
            A array of shape `(3,)` giving the cartesian coordinates of the centre of
            grid square at index `[i, j]` plus the time step of the sheet.

        """
        x_step = (self.extent[1] - self.extent[0]) / resolution[0]
        y_step = (self.extent[3] - self.extent[2]) / resolution[1]
        return np.array([
            self.extent[0] + (j + 0.5) * x_step,
            self.extent[3] - (i + 0.5) * y_step,
            k * self.t_max / resolution[-1]
        ])

    def signal_distribution(
            self,
            resolution: tuple[int, int, int],
            sensor_indices: tuple[int, ...] | None = None,
            source_indices: tuple[int, ...] | None = None,
    ) -> np.ndarray:
        """
        Calculates the log posterior distribution for the location of the sources.

        Parameters
        ----------
        resolution : tuple[int, int, int]
            The grid size (#rows, #columns, #frames), in which to break up the extent.
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

        log_dist = np.zeros(permute21(resolution))

        def time_to_t_idx(time):
            return int(time * resolution[-1] / self.t_max)

        for sensor, signal in zip(
                self.sensors[sensor_indices, :],
                self.signals[np.ix_(sensor_indices, source_indices)]
        ):
            for idx, _ in np.ndenumerate(log_dist):
                i, j, k = idx
                location = self._idx_to_loc(i, j, k, resolution)
                xy_loc = location[:2]
                t = location[-1]

                expected_time = t + sim.dist(sensor, xy_loc) / self.signal_speed
                mean_shifted = signal - expected_time
                log_vals = -0.5 * np.square(mean_shifted / self.noise)
                log_dist[idx] += np.logaddexp.reduce(log_vals)

        log_dist -= np.logaddexp.reduce(log_dist, axis=None)
        return log_dist

    def compute_distribution(self, resolution: tuple[int, int, int]) -> None:
        """
        Compute the log posterior distribution for all data and save data.

        Parameters
        ----------
        resolution : tuple[int, int, int]
            The grid size (#rows, #columns, #frames), in which to break up the extent.

        Returns
        -------
        None

        """
        self.log_distribution = self.signal_distribution(resolution=resolution)
        self._dist_calculated = True

    def save_distribution(self, file: str) -> str:
        """
        Saves the distribution to a file.

        Parameters
        ----------
        file : the filename

        Returns
        -------
        str
            The filename

        """
        self._log_dist_error()
        np.save(file=file, arr=self.log_distribution)
        return file

    def load_distribution(self, file: str) -> None:
        """
        Loads the log distribution from a file.

        Parameters
        ----------
        file : str
            The filename.
        """
        self.log_distribution = np.load(file=file)
        self._dist_calculated = True

    def maxima_locs(self) -> np.ndarray:
        """
        Calculate the location of local maxima in the log posterior distribution.

        Returns
        -------
        np.ndarray
            The cartesian coordinates of the local maxima with time in the third column.

        """
        self._log_dist_error()
        resolution = permute21(self.log_distribution.shape)
        maxima_idx = maxima(self.log_distribution, self.sources.shape[0])
        return np.array([
            self._idx_to_loc(i, j, k, resolution) for i, j, k in maxima_idx
        ])

    def display(
            self,
            exp_factor: float = 1e-4,
            show_sources: bool = True,
            show_maxima: bool = False,
            flatten: bool = False,
            fps: int = 10
    ):
        """
        Displays the posterior distribution.

        Parameters
        ----------
        exp_factor : float, optional
            A factor with which to scale the log posterior before taking the
            exponential. Useful to get sensible pictures (default is 1e-4).
        show_sources : bool, optional
            If `True`, show the true location of the sources (default is `True`)
        show_maxima : bool, optional
            If `True`, show the calculated maxima in the posterior, i.e. the most
            likely locations of the sources according to the posterior (default is
            `False`).
        flatten : bool, optional
            If `True` the posterior is flattened to a single frame by taking the
            maximum along the time axis (a more rigorous approach may be implemented
            later) (default is `False`).
        fps : int, optional
            Frames per second of the animation. Default is 10.

        Returns
        -------
        matplotlib.Figure, matplotlib.Axes

        """

        self._log_dist_error()
        ar = sim.SensorField(*self.extent, sensor_locations=self.sensors)
        fig, ax = ar.display()

        if flatten:
            dist = np.max(self.log_distribution, axis=2)
            dist = dist[..., np.newaxis]
        else:
            dist = self.log_distribution

        a = np.exp(exp_factor * dist)
        norm = Normalize(vmin=np.min(a), vmax=np.max(a))

        if dist.shape[-1] == 1:
            ax.imshow(
                a,
                cmap=MAGMA_ALPHAS_cmap,
                interpolation='nearest',
                extent=self.extent,
                zorder=50,
                #alpha=a/np.max(a),
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

        num_frames = dist.shape[-1]

        im = ax.imshow(
            a[..., 0],
            cmap=MAGMA_ALPHAS_cmap,
            norm=norm,
            interpolation='nearest',
            extent=self.extent,
            zorder=50,
        )
        source_sc = ax.scatter(
            [],
            [],
            c='limegreen',
            marker='*',
            zorder=100,
            s=100,
        )
        maxima_sc = ax.scatter(
            [],
            [],
            c='red',
            marker='+',
            zorder=100,
            s=100,
        )

        event_duration = 1000  # in milliseconds
        event_frames = int((event_duration * fps / 1000) / 2) * 2

        src_events = []
        for idx, source in enumerate(self.sources):
            src_frame = int(num_frames * source[-1] / self.t_max)
            start = src_frame - event_frames / 2
            end = src_frame + event_frames / 2
            src_events.append((start, end, idx))

        max_locs = self.maxima_locs()
        max_events = []
        for idx, maxima in enumerate(max_locs):
            max_frame = int(num_frames * maxima[-1] / self.t_max)
            start = max_frame - event_frames / 2
            end = max_frame + event_frames / 2
            max_events.append((start, end, idx))

        def resizer(frame, e):
            f = 1 - abs((2 * frame - e[0] - e[1]) / (e[0] - e[1]))
            return f

        def update(frame):
            im.set_data(a[..., frame])

            frame_src_events = tuple(e for e in src_events if (e[0] <= frame <= e[1]))
            frame_max_events = tuple(e for e in max_events if (e[0] <= frame <= e[1]))

            frame_src_marks = tuple(e[-1] for e in frame_src_events)
            frame_max_marks = tuple(e[-1] for e in frame_max_events)

            source_sc.set_offsets(self.sources[frame_src_marks, :2])
            maxima_sc.set_offsets(max_locs[frame_max_marks, :2])

            frame_src_alphas = tuple(
                 resizer(frame, e) for e in frame_src_events
            )
            frame_max_alphas = tuple(
                 resizer(frame, e) for e in frame_max_events
            )

            if len(frame_src_events) > 0:
                # This and the below if block are here as the set_alpha method throws an
                # error if given an empty tuple - even if the set of items in the
                # collection is empty.
                source_sc.set_alpha(frame_src_alphas)
            if len(frame_max_events) > 0:
                maxima_sc.set_alpha(frame_max_alphas)

            source_sc.set_sizes(tuple(200 * f for f in frame_src_alphas))
            maxima_sc.set_sizes(tuple(200 * f for f in frame_max_alphas))

            return [im, source_sc, maxima_sc]

        anim = animation.FuncAnimation(
            fig=fig,
            func=update,
            frames=num_frames,
            interval=1000 / fps
        )
        plt.close()
        return anim

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
