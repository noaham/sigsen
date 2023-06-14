"""
Module providing utilities for the purpose of simulating data collected by sensors.
"""

import numpy as np
from dataclasses import dataclass


@dataclass
class SensorData:
    """
    A simple container class representing the sensor data from a single event
    """
    sensors: np.ndarray
    sources: np.ndarray
    signals: np.ndarray


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
            sensor_locations: list[tuple[float, float]] | np.ndarray = None,
    ) -> None:
        """
        random, grid, specified
        """
        self.xmin = xmin
        self.xmax = xmax
        self.ymin = ymin
        self.ymax = ymax
        self.sensors = np.array((0, 2))
        self._name_counter = 0

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

    def display(self) -> None:
        pass

    def gen_data(
            self,
            num_sources: int = 1,
            source_locs: np.ndarray | None = None,
            signal_speed: float = 1,
            noise: float = 0.1,
    ) -> SensorData:
        if source_locs is None:
            source_locs = self._random_locations(num_sources)
        signals = np.sqrt(np.matmul(self.sensors, source_locs)) / signal_speed
        signals += self.rng.normal(0, noise, signals.shape)
        return SensorData(self.sensors, source_locs, signals)
