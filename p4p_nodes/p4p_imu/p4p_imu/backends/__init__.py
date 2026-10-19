"""IMU backends: one interface, two implementations, chosen by parameter.

The node talks only to ImuBackend, so the code that cannot be tested without the
sensor is confined to bno08x.py. mock.py exists so that the whole node -- the
validity gate, the staleness watchdog, the reset counter, the reconnect loop, QoS
negotiation -- is exercised by the test suite and can be brought up on a bench
with no sensor attached.
"""
from __future__ import annotations

import abc
import importlib

from ..orientation import ImuSample

BACKENDS = {
    'bno08x': ('.bno08x', 'Bno08xBackend'),
    'mock': ('.mock', 'MockBackend'),
}


class ImuError(RuntimeError):
    """Error raised when the sensor cannot be opened, configured or read."""


class ImuResetError(ImuError):
    """Error raised when the sensor resets itself and can be recovered in place."""


class ImuBackend(abc.ABC):
    """Abstract interface to one IMU, called only from the node's read thread."""

    name: str = 'backend'

    @abc.abstractmethod
    def start(self) -> None:
        """Open the bus, configure the sensor, and wait for the first valid sample."""

    @abc.abstractmethod
    def read(self) -> ImuSample:
        """Return one coherent sample: all three vectors from a single instant."""

    @abc.abstractmethod
    def recover(self) -> None:
        """Re-configure a sensor that reset itself, without rebuilding the bus."""

    @abc.abstractmethod
    def stop(self) -> None:
        """Release the sensor and the bus.

        Idempotent, and must not raise.
        """


def make_backend(name: str, params: dict, logger) -> ImuBackend:
    """Instantiate a backend by name, importing its module lazily.

    Args:
        name: Backend name, a key of BACKENDS (bno08x or mock).
        params: Backend settings gathered from the node's ROS parameters.
        logger: rclpy logger the backend reports through.
    """
    try:
        module_name, class_name = BACKENDS[name]
    except KeyError:
        raise ImuError(
            f'unknown imu_backend {name!r}; expected one of {sorted(BACKENDS)}') from None
    module = importlib.import_module(module_name, __package__)
    return getattr(module, class_name)(params, logger)
