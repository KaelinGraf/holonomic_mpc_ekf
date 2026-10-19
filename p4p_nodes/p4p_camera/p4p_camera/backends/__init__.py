"""Camera backends: one interface, three implementations, chosen by parameter.

The node talks only to CameraBackend, so the code that cannot be tested without
the sensor is confined to picam.py and v4l2.py. mock.py exists so that the whole
node -- geometry, encoding, the service, QoS negotiation -- is exercised by the
test suite and can be brought up on a bench with no camera attached.
"""
from __future__ import annotations

import abc
import importlib

import numpy as np

BACKENDS = {
    'picamera2': ('.picam', 'Picamera2Backend'),
    'v4l2': ('.v4l2', 'V4L2Backend'),
    'mock': ('.mock', 'MockBackend'),
}


class CameraError(RuntimeError):
    """Error raised when the camera cannot be opened, configured or read."""


class CameraBackend(abc.ABC):
    """Interface to one camera, owned by one thread.

    Only the node's capture thread ever calls start, read or stop.
    """

    name: str = 'backend'

    @abc.abstractmethod
    def start(self) -> None:
        """Open and configure the camera, and begin streaming, raising CameraError on failure."""

    @abc.abstractmethod
    def read(self, timeout_s: float) -> tuple[np.ndarray, int | None]:
        """Return one frames.FULL uint8 frame and its capture time.

        The frame is a copy the caller owns; the time is CLOCK_BOOTTIME [ns], or None.

        Args:
            timeout_s: Longest wait for a frame before TimeoutError [s].
        """

    @abc.abstractmethod
    def stop(self) -> None:
        """Release the camera; idempotent, and never raises."""


def make_backend(name: str, params: dict, logger) -> CameraBackend:
    """Instantiate a backend by name, importing its module lazily.

    Args:
        name: Backend name, a key of BACKENDS.
        params: Backend settings collected from the node's parameters.
        logger: rclpy logger for the backend to report through.
    """
    try:
        module_name, class_name = BACKENDS[name]
    except KeyError:
        raise CameraError(
            f'unknown camera_backend {name!r}; expected one of '
            f'{sorted(BACKENDS)}') from None
    module = importlib.import_module(module_name, __package__)
    return getattr(module, class_name)(params, logger)
