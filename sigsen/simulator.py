"""
Module providing utilities for the purpose of simulating data collected by sensors.
"""

import numpy as np
import matplotlib.pyplot as plt
import sigsen.analysis as an
plt.style.use("seaborn")


def dist(p: np.ndarray, q: np.ndarray) -> float:
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
        random, grid, specified
        """
        self.xmin = xmin
        self.xmax = xmax
        self.ymin = ymin
        self.ymax = ymax
        self.sensors = sensor_locations

        self.add_sensors(sensor_locations)

        self.rng = np.random.default_rng()

    def _is_in_range(self, x: float, y: float) -> bool:
        return all([
            self.xmin <= x,
            x <= self.xmax,
            self.ymin <= y,
            y <= self.ymax,
        ])

    def add_sensors(self, sensors: list | np.ndarray) -> None:
        sensors = np.array(sensors)
        if sensors.shape == (2,):
            sensors.reshape((1, 2))
        elif len(sensors.shape) != 2 or sensors.shape[1] != 2:
            raise ValueError(f'The sensors to be added has shape {sensors.shape}')
        if not all(self._is_in_range(*row) for row in sensors):
            raise ValueError(f'Some rows represent points outside the range.')
        self.sensors = np.concatenate((self.sensors, sensors), axis=0)

    def _random_locations(self, num: int = 1) -> np.ndarray:
        ranges = (self.xmax - self.xmin, self.ymax - self.ymin)
        return self.rng.random((num, 2)) * ranges + (self.xmin, self.ymin)

    def add_random_sensors(self, num: int = 1) -> None:
        self.add_sensors(self._random_locations(num))

    def add_sensor_grid(
            self,
            shape: tuple[int, int],
            extent: tuple[float, ...] | None = None,
    ) -> None:
        if extent is None:
            extent = (self.xmin, self.xmax, self.ymin, self.ymax)

        xstep = (extent[1] - extent[0]) / (shape[1] - 1)
        ystep = (extent[3] - extent[2]) / (shape[0] - 1)

        sensors = np.array([
            [
                extent[0] + xstep * i,
                extent[2] + ystep * j,
            ] for i in range(shape[1]) for j in range(shape[0])
        ])

        self.add_sensors(sensors)

    def display(self) -> None:
        fig, ax = plt.subplots()
        ax.scatter(self.sensors[:, 0], self.sensors[:, 1])
        plt.show()

    def gen_data(
            self,
            num_sources: int = 1,
            source_locs: np.ndarray | None = None,
            signal_speed: float = 1,
            noise: float = 0.1,
    ) -> an.SensorData:
        if source_locs is None:
            source_locs = self._random_locations(num_sources)

        signals = np.array([
            [
                dist(source, sensor).reshape(()) / signal_speed for source in source_locs
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
