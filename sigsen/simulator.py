"""
Module providing utilities for the purpose of simulating data collected by sensors.
"""

import numpy as np
from matplotlib.patches import Rectangle
import matplotlib.pyplot as plt
import sigsen.analysis as an
plt.style.use("seaborn")


def dist(p: np.ndarray, q: np.ndarray) -> np.ndarray:
    """
    The euclidian distance between lists of vectors. `p` and `q` must be arrays of
    the same shape and are interpreted as lists of horizontal vectors. The result is
    a list representing the distance between a row of `p` and the corresponding row
    of `q`.

    Parameters
    ----------
    p : np.ndarray
        An array
    q : np.ndarray
        An array

    Returns
    -------
    np.ndarray
        An array containing the distances between corresponding rows of the input
        arrays.

    """
    if p.shape != q.shape:
        raise TypeError(f'Inputs shapes {p.shape} and {q.shape} are not equal.')
    if len(p.shape) == 1:
        p = p.reshape((1, p.shape[0]))
    if len(p.shape) > 2:
        p = p.reshape((p.shape[0], sum(i for i in p.shape[1:])))
    return np.sqrt(np.sum(np.square(p-q), axis=1))


class SensorField:
    """
    A class representing a sensor field from which simulated data may be generated.
    """
    def __init__(
            self,
            xmin: float,
            xmax: float,
            ymin: float,
            ymax: float,
            sensor_locations: list[tuple[float, float]] | np.ndarray = np.empty((0, 2)),
    ) -> None:
        """

        Parameters
        ----------
        xmin : float
            The minimum extent in the x direction.
        xmax : float
            The maximum extent in the x direction.
        ymin : float
            The minimum extent in the y direction.
        ymax : float
            The maximum extent in the y direction.
        sensor_locations : list[tuple[float, float]] | np.ndarray = np.empty((0, 2))
            (Optional) The location of the sensors given as a 2D array with
            x-coordinate in the first column and y-coordinate in the second column (the
            default is an empty array to which sensor locations can be added later).
        """
        self.xmin = xmin
        self.xmax = xmax
        self.ymin = ymin
        self.ymax = ymax
        self.sensors = sensor_locations
        self.add_sensors(sensor_locations)
        self.rng = np.random.default_rng()

    def _is_in_range(self, x: float, y: float) -> bool:
        """
        Given an x and y coordinate return `True` if this represents a point within
        the extent of the sensor field..

        Parameters
        ----------
        x : float
            The x-coordinate
        y : float
            The y-coordinate

        Returns
        -------
        bool
            `True` if `(x,y)` represents a point within the rectangle bounding the
            sensor field. Otherwise `False`.

        """
        return all([
            self.xmin <= x,
            x <= self.xmax,
            self.ymin <= y,
            y <= self.ymax,
        ])

    def add_sensors(self, sensors: list | np.ndarray) -> None:
        """
        Add sensors at locations specified in a list or array.

        Parameters
        ----------
        sensors : list | np.ndarray
            The list or array containing sensor locations as `(x,y)` coordinate pairs.

        Returns
        -------
        None

        """
        sensors = np.array(sensors)
        if sensors.shape == (2,):
            sensors.reshape((1, 2))
        elif len(sensors.shape) != 2 or sensors.shape[1] != 2:
            raise ValueError(f'The sensors to be added has shape {sensors.shape}')
        if not all(self._is_in_range(*row) for row in sensors):
            raise ValueError(f'Some rows represent points outside the range.')
        self.sensors = np.concatenate((self.sensors, sensors), axis=0)

    def _random_locations(self, num: int = 1) -> np.ndarray:
        """
        Generates a list of locations within the sensor field's bounding box chosen
        uniformly at random.

        Parameters
        ----------
        num : int, optional
            The number of locations to generate (default is 1)

        Returns
        -------
        np.ndarray
            The random locations as `(x,y)` pairs.

        """
        ranges = (self.xmax - self.xmin, self.ymax - self.ymin)
        return self.rng.random((num, 2)) * ranges + (self.xmin, self.ymin)

    def add_random_sensors(self, num: int = 1) -> None:
        """
        Add a number of sensors at random locations.

        Parameters
        ----------
        num : int, optional
            The number of sensors to add (default is 1).

        Returns
        -------
        None

        """
        self.add_sensors(self._random_locations(num))

    def add_sensor_grid(
            self,
            shape: tuple[int, int],
            extent: tuple[float, ...] | None = None,
    ) -> None:
        """
        Add a grid of sensors.

        Parameters
        ----------
        shape : tuple[int, int]
            A tuple `(i,j)` representing the number of rows and columns in the grid
            respectively.
        extent : tuple[float, ...] | None, optional
            A tuple `(xmin, xmax, ymin, ymax)` giving the bounding box over which the
            grid will be spread (default is `None` which causes the bounding box of
            the sensor field to be used.

        Returns
        -------
        None

        """
        if extent is None:
            extent = (self.xmin, self.xmax, self.ymin, self.ymax)

        xstep = (extent[1] - extent[0]) / shape[1]
        ystep = (extent[3] - extent[2]) / shape[0]

        sensors = np.array([
            [
                extent[0] + xstep * (i+0.5),
                extent[2] + ystep * (j+0.5),
            ] for i in range(shape[1]) for j in range(shape[0])
        ])

        self.add_sensors(sensors)

    def display(self):
        """
        Displays a pictorial representation of the sensor field and the sensor
        locations.

        Returns
        -------
        matplotlib.Figure, matplotlib.Axes
            The figure and axes objects being plotted to.

        """
        width, height = self.xmax - self.xmin, self.ymax - self.ymin
        bf = 0.1
        xmargin = bf * width
        ymargin = bf * height

        fig, ax = plt.subplots()
        rectangle = Rectangle(
            (self.xmin, self.ymin),
            width,
            height,
            color='grey',
            alpha=0.05,
            linewidth=1,
            zorder=2,
        )
        ax.add_patch(rectangle)

        ax.scatter(
            self.sensors[:, 0],
            self.sensors[:, 1],
            zorder=100,
            c='blue',
            marker='.'
        )

        xlims = [self.xmin - xmargin, self.xmax + xmargin]
        ylims = [self.ymin - ymargin, self.ymax + ymargin]

        ax.set_xlim(*xlims)
        ax.set_ylim(*ylims)
        return fig, ax

    def gen_data(
            self,
            time_span: float = 0,
            num_sources: int = 1,
            source_locs: np.ndarray | None = None,
            signal_speed: float = 1,
            noise: float = 0.1,
    ) -> an.SensorData:
        """
        Generates sensor data given signal sources at specified (or random) locations.

        Parameters
        ----------
        time_span: float
            The time span over which to simulate source events (default is zero which
            means all source events are simultaneous).
        num_sources : int, optional
            The number of signal sources (default is 1).
        source_locs : np.ndarray | None, optional
            The source locations given as an array of `(x,y)` pairs if `time_span` is
            zero, or `(x,y,t)` triples which additionally indicate the time of the
            source (default is `None` which causes random source locations to be used).
        signal_speed : float, optional
            The speed at which signals travel (default is 1).
        noise : float, optional
            The level of noise in the data generated (default is 0.1).

        Returns
        -------
        sigsen.SensorData
            A `SensorData` object representing the data collected by the sensors.

        """
        if source_locs is None:
            source_locs = self._random_locations(num_sources)
        if source_locs.shape[2] == 2:
            source_locs = np.concatenate(
                (source_locs, time_span * self.rng.random((num_sources, 1))),
                axis=1,
            )

        signals = np.array([
            [
                source[2] + dist(source[:2], sensor).reshape(()) / signal_speed
                for source in source_locs
            ] for sensor in self.sensors
        ]) + self.rng.normal(0, noise, (len(self.sensors), num_sources))

        return an.SensorData(
            sensors=self.sensors,
            sources=source_locs,
            signals=signals,
            extent=(self.xmin, self.xmax, self.ymin, self.ymax),
            signal_speed=signal_speed,
            noise=noise
        )
