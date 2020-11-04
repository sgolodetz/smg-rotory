import numpy as np

from abc import ABC, abstractmethod


class Drone(ABC):
    """The abstract base class for drone interfaces."""

    # PUBLIC ABSTRACT METHODS

    @abstractmethod
    def get_image(self) -> np.ndarray:
        """
        Get the most recent image received from the drone.

        :return:    The most recent image received from the drone.
        """
        pass

    @abstractmethod
    def land(self):
        """Tell the drone to land."""
        pass

    @abstractmethod
    def takeoff(self):
        """Tell the drone to take off."""
        pass
