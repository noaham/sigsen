"""
Module providing statistical modelling functionality
"""

import numpy as np
import sigsen.simulator as sim
from dataclasses import dataclass


@dataclass
class SensorData:
    """
    A simple container class representing the sensor data from a single event
    """
    sensors: np.ndarray
    sources: np.ndarray
    signals: np.ndarray

    extent: tuple[float, ...]
    signal_speed: float
    noise: float

    log_distribution: np.ndarray | None = None
    
    def _log_dist_error(self):
        if self.log_distribution is None:
            raise ValueError('The distribution has not yet been calculated.')

    def signal_distribution(
            self,
            resolution: tuple[int, int],
            sensor_indices: tuple[int, ...] | None = None,
            source_indices: tuple[int, ...] | None = None,
    ) -> np.ndarray:
        if sensor_indices is None:
            sensor_indices = tuple(range(len(self.sensors)))
        if source_indices is None:
            source_indices = tuple(range(len(self.sources)))

        log_dist = np.zeros(resolution)
        x_step = (self.extent[1] - self.extent[0]) / resolution[1]
        y_step = (self.extent[3] - self.extent[2]) / resolution[0]

        for sensor, signal in zip(
                self.sensors[sensor_indices, :],
                self.signals[np.ix_(sensor_indices, source_indices)]
        ):
            for idx, d in np.ndenumerate(log_dist):
                i, j = idx
                location = np.array([
                    self.extent[0] + (j + 0.5) * x_step,
                    self.extent[3] - (i + 0.5) * y_step
                ])

                mean_shifted = signal - sim.dist(sensor, location)
                log_vals = -0.5 * np.square(mean_shifted / self.noise)
                log_dist[idx] += np.logaddexp.reduce(log_vals)

        log_dist -= np.logaddexp.reduce(log_dist, axis=None)
        return log_dist

    def compute_distribution(self, resolution: tuple[int, int]) -> None:
        self.log_distribution = self.signal_distribution(resolution=resolution)

    def display(self, exp_factor: float = 0.01):
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

        ax.scatter(
            x=self.sources[:, 0],
            y=self.sources[:, 1],
            c='limegreen',
            marker='*',
            zorder=100,
            s=100,
        )

        return fig, ax
