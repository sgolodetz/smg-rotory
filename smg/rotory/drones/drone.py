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
    def land(self) -> None:
        """Tell the drone to land."""
        pass

    @abstractmethod
    def move_forward(self, rate: float) -> None:
        """
        Tell the drone to move forward at the specified rate.

        This can also be used to move backwards (by specifying a negative rate).

        :param rate:     The rate at which the drone should move forward (in [-1,1]).
        """
        pass

    @abstractmethod
    def move_right(self, rate: float) -> None:
        """
        Tell the drone to move to the right at the specified rate.

        This can also be used to move to the left (by specifying a negative rate).

        :param rate:    The rate at which the drone should move to the right (in [-1,1]).
        """
        pass

    @abstractmethod
    def move_up(self, rate: float) -> None:
        """
        Tell the drone to move up at the specified rate.

        This can also be used to move down (by specifying a negative rate).

        :param rate:    The rate at which the drone should move up (in [-1,1]).
        """
        pass

    @abstractmethod
    def stop(self) -> None:
        """Tell the drone to stop in mid-air."""
        pass

    @abstractmethod
    def takeoff(self) -> None:
        """Tell the drone to take off."""
        pass

    @abstractmethod
    def turn(self, rate: float) -> None:
        """
        Tell the drone to turn at the specified rate.

        :param rate:    The rate at which the drone should turn (in [-1,1]).
        """
        pass
