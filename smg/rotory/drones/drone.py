import numpy as np

from abc import ABC, abstractmethod
from typing import Dict, Optional, Tuple

from ..util.beacon import Beacon


class Drone(ABC):
    """The abstract base class for drone interfaces."""

    # NESTED TYPES

    class EState(int):
        """The states in which a drone can be."""
        pass

    # The drone is on the ground, with its motors switched off.
    IDLE: EState = EState(0)

    # The drone is in the process of performing an automated take-off.
    TAKING_OFF: EState = EState(1)

    # The drone is flying normally.
    FLYING: EState = EState(2)

    # The drone is in the process of performing an automated landing.
    LANDING: EState = EState(3)

    # CONSTRUCTOR

    def __init__(self):
        """TODO"""
        self.__localised_beacons: Dict[str, Beacon] = {}

    # PUBLIC ABSTRACT METHODS

    @abstractmethod
    def get_battery_level(self) -> Optional[int]:
        """
        Try to get the most recently received value of the remaining battery %.

        :return:    The most recently received value of the remaining battery %, if available, or None otherwise.
        """
        pass

    @abstractmethod
    def get_image(self) -> np.ndarray:
        """
        Get the most recent image received from the drone.

        :return:    The most recent image received from the drone.
        """
        pass

    @abstractmethod
    def get_image_size(self) -> Tuple[int, int]:
        """
        Get the size of the images captured by the drone.

        :return:    The size of the images captured by the drone, as a (width, height) tuple.
        """
        pass

    @abstractmethod
    def get_intrinsics(self) -> Optional[Tuple[float, float, float, float]]:
        """
        Get the camera intrinsics, if known.

        :return:    The camera intrinsics as an (fx, fy, cx, cy) tuple, if known, or None otherwise.
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
    def terminate(self) -> None:
        """Tell the drone to terminate."""
        pass

    @abstractmethod
    def turn(self, rate: float) -> None:
        """
        Tell the drone to turn at the specified rate.

        :param rate:    The rate at which the drone should turn (in [-1,1]).
        """
        pass

    # PUBLIC METHODS

    def get_beacon_ranges(self, drone_pos: Optional[np.ndarray] = None, *, include_localised: bool = True,
                          include_unlocalised: bool = True) -> Dict[str, float]:
        """
        Get the estimated ranges (in m) between the drone and any beacons that are within range.

        .. note::
            The number of ranges returned may vary over time.

        :param drone_pos:           The current position of the drone (if available).
        :param include_localised:   TODO
        :param include_unlocalised: TODO
        :return:                    A dictionary that maps the names of the beacons to their estimated ranges (in m).
        """
        beacon_ranges: Dict[str, float] = {}

        if include_localised and drone_pos is not None:
            for beacon_name, beacon in self.__localised_beacons.items():
                beacon_range: float = np.linalg.norm(beacon.position - drone_pos)
                if beacon_range <= beacon.max_range:
                    beacon_ranges[beacon_name] = beacon_range

        if include_unlocalised:
            # TODO: Unlocalised beacons.
            pass

        return beacon_ranges

    # noinspection PyMethodMayBeStatic
    def get_expected_takeoff_height(self) -> Optional[float]:
        """
        Try to get the height (in m) to which the drone is expected to take off (if known).

        :return:    The height (in m) to which the drone is expected to take off, if known, or None otherwise.
        """
        return None

    def get_height(self) -> Optional[float]:
        """
        Try to get the drone's height (in m).

        :return:    The most recently received value of the drone's height (in m), if available, or None otherwise.
        """
        return None

    def get_localised_beacons(self) -> Dict[str, Beacon]:
        """
        TODO

        :return:    TODO
        """
        return self.__localised_beacons.copy()

    def get_state(self) -> Optional[EState]:
        """
        Try to get the current state of the drone.

        :return:    The current state of the drone, if known, or None otherwise.
        """
        return None

    def get_timed_image(self) -> Tuple[np.ndarray, Optional[float]]:
        """
        Get the most recent image received from the drone and its (optional) timestamp.

        :return:    A pair consisting of the most recent image received from the drone and its (optional) timestamp.
        """
        return self.get_image(), None

    def set_localised_beacon(self, beacon_name: str, beacon: Optional[Beacon]) -> None:
        """
        TODO

        :param beacon_name: TODO
        :param beacon:      TODO
        """
        if beacon is not None:
            self.__localised_beacons[beacon_name] = beacon
        else:
            del self.__localised_beacons[beacon_name]

    def update_gimbal_pitch(self, gimbal_pitch: float) -> None:
        """
        Update the pitch of the drone's gimbal (if it has one that can be pitched so as to look up/down).

        .. note::
            If the drone doesn't have a suitable gimbal, this will be a no-op.

        :param gimbal_pitch:   The desired new pitch for the drone's gimbal (in [-1,1], where -1 = down and 1 = up).
        """
        pass
