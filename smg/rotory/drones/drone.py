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

    def calculate_forward_rate(self, m_per_s: float, *, allow_clipping: bool = True) -> Optional[float]:
        """
        Try to calculate a rate in [-1,1] that would move the drone forward or backward with the specified
        velocity (in m/s), if one exists.

        .. note::
            The specified velocity may not be achievable in practice, e.g. due to physical limitations of the drone.
            This will typically result in a "raw" rate that is outside the [-1,1] range. If that happens, the function
            will either clip the "raw" rate to one that is in range, if allow_clipping is set to True, or return None
            if it's set to False.
        .. note::
            In terms of signs, a +ve velocity means "move forward" and a -ve velocity means "move backward".
            The calculated rate will be +ve when moving forward and -ve when moving backward.

        :param m_per_s:         The velocity (in m/s).
        :param allow_clipping:  Whether to allow the "raw" rate to be clipped to the [-1,1] range.
        :return:                The corresponding "raw" rate (in m/s), if it's in range, else the result of clipping
                                it to the [-1,1] range if clipping is allowed, else None.
        """
        return None

    def calculate_forward_velocity(self, rate: float) -> Optional[float]:
        """
        Calculate the velocity (in m/s) at which the specified rate (in [-1,1]) will move the drone forward
        or backward.

        .. note::
            In terms of signs, a rate of 1.0 means "move forward at maximum speed" and a rate of -1.0
            means "move backward at maximum speed". The velocity will be +ve when moving forward and
            -ve when moving backward.

        :param rate:    The rate (in [-1,1]).
        :return:        The corresponding velocity (in m/s).
        """
        return None

    def calculate_right_rate(self, m_per_s: float, *, allow_clipping: bool = True) -> Optional[float]:
        """
        Try to calculate a rate in [-1,1] that would move the drone right or left with the specified velocity (in m/s),
        if one exists.

        .. note::
            The specified velocity may not be achievable in practice, e.g. due to physical limitations of the drone.
            This will typically result in a "raw" rate that is outside the [-1,1] range. If that happens, the function
            will either clip the "raw" rate to one that is in range, if allow_clipping is set to True, or return None
            if it's set to False.
        .. note::
            In terms of signs, a +ve velocity means "move right" and a -ve velocity means "move left".
            The calculated rate will be +ve when moving right and -ve when moving left.

        :param m_per_s:         The velocity (in m/s).
        :param allow_clipping:  Whether to allow the "raw" rate to be clipped to the [-1,1] range.
        :return:                The corresponding "raw" rate (in m/s), if it's in range, else the result of clipping
                                it to the [-1,1] range if clipping is allowed, else None.
        """
        return None

    def calculate_right_velocity(self, rate: float) -> Optional[float]:
        """
        Calculate the velocity (in m/s) at which the specified rate (in [-1,1]) will move the drone right or left.

        .. note::
            In terms of signs, a rate of 1.0 means "move right at maximum speed" and a rate of -1.0
            means "move left at maximum speed". The velocity will be +ve when moving right and -ve
            when moving left.

        :param rate:    The rate (in [-1,1]).
        :return:        The corresponding velocity (in m/s).
        """
        return None

    def calculate_turn_rate(self, rad_per_s: float, *, allow_clipping: bool = True) -> Optional[float]:
        """
        Try to calculate a rate in [-1,1] that would turn the drone right or left with the specified
        angular velocity (in rad/s), if one exists.

        .. note::
            The specified angular velocity may not be achievable in practice, e.g. due to physical limitations of the
            drone. This will typically result in a "raw" rate that is outside the [-1,1] range. If that happens, the
            function will either clip the "raw" rate to one that is in range, if allow_clipping is set to True, or
            return None if it's set to False.
        .. note::
            In terms of signs, a +ve angular velocity means "turn left" and a -ve velocity means "turn right".
            The calculated rate will be +ve when turning right and -ve when turning left. Note that we use
            different signs for the velocities and the rates, as angles are usually measured anti-clockwise
            around a circle.

        :param rad_per_s:       The angular velocity (in rad/s).
        :param allow_clipping:  Whether to allow the "raw" rate to be clipped to the [-1,1] range.
        :return:                The corresponding "raw" rate (in m/s), if it's in range, else the result of clipping
                                it to the [-1,1] range if clipping is allowed, else None.
        """
        return None

    def calculate_turn_velocity(self, rate: float) -> Optional[float]:
        """
        Calculate the angular velocity (in rad/s) at which the specified rate (in [-1,1]) will turn the drone
        right or left.

        .. note::
            In terms of signs, a rate of 1.0 means "turn right at maximum speed" and a rate of -1.0
            means "turn left at maximum speed". The angular velocity will be -ve when turning right,
            and +ve when turning left, as angles are usually measured anti-clockwise around a circle.

        :param rate:    The rate (in [-1,1]).
        :return:        The corresponding angular velocity (in rad/s).
        """
        return None

    def calculate_up_rate(self, m_per_s: float, *, allow_clipping: bool = True) -> Optional[float]:
        """
        Try to calculate a rate in [-1,1] that would move the drone up or down with the specified velocity (in m/s),
        if one exists.

        .. note::
            The specified velocity may not be achievable in practice, e.g. due to physical limitations of the drone.
            This will typically result in a "raw" rate that is outside the [-1,1] range. If that happens, the function
            will either clip the "raw" rate to one that is in range, if allow_clipping is set to True, or return None
            if it's set to False.
        .. note::
            In terms of signs, a +ve velocity means "move up" and a -ve velocity means "move down".
            The calculated rate will be +ve when moving up and -ve when moving down.

        :param m_per_s:         The velocity (in m/s).
        :param allow_clipping:  Whether to allow the "raw" rate to be clipped to the [-1,1] range.
        :return:                The corresponding "raw" rate (in m/s), if it's in range, else the result of clipping
                                it to the [-1,1] range if clipping is allowed, else None.
        """
        return None

    def calculate_up_velocity(self, rate: float) -> Optional[float]:
        """
        Calculate the velocity (in m/s) at which the specified rate (in [-1,1]) will move the drone up or down.

        .. note::
            In terms of signs, a rate of 1.0 means "move up at maximum speed" and a rate of -1.0
            means "move down at maximum speed". The velocity will be +ve when moving up and -ve
            when moving down.

        :param rate:    The rate (in [-1,1]).
        :return:        The corresponding velocity (in m/s).
        """
        return None

    def clip_forward_velocity(self, m_per_s: float) -> float:
        """
        Clip the specified forward/backward velocity (in m/s) to one that is achievable by the drone.

        :param m_per_s: The input velocity (in m/s).
        :return:        The result of clipping the input velocity to one that is achievable by the drone.
        """
        return self.calculate_forward_velocity(self.calculate_forward_rate(m_per_s))

    def clip_right_velocity(self, m_per_s: float) -> float:
        """
        Clip the specified right/left velocity (in m/s) to one that is achievable by the drone.

        :param m_per_s: The input velocity (in m/s).
        :return:        The result of clipping the input velocity to one that is achievable by the drone.
        """
        return self.calculate_right_velocity(self.calculate_right_rate(m_per_s))

    def clip_up_velocity(self, m_per_s: float) -> float:
        """
        Clip the specified up/down velocity (in m/s) to one that is achievable by the drone.

        :param m_per_s: The input velocity (in m/s).
        :return:        The result of clipping the input velocity to one that is achievable by the drone.
        """
        return self.calculate_up_velocity(self.calculate_up_rate(m_per_s))

    def get_beacon_ranges(self, *, drone_pos: Optional[np.ndarray] = None,
                          test_beacons: Optional[Dict[str, Beacon]] = None) -> Dict[str, float]:
        """
        Get the estimated ranges (in m) between the drone and any beacons that are within range.

        .. note::
            The number of ranges returned may vary over time.

        :param drone_pos:       The current position of the drone (if available).
        :param test_beacons:    TODO
        :return:                A dictionary that maps the names of the beacons to their estimated ranges (in m).
        """
        return {}

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

    def has_calibrated_rates(self) -> bool:
        """
        Get whether or not this type of drone has had its rate <-> velocity conversion functions implemented yet.

        :return:    True, if the rate <-> velocity conversion functions have been implemented for this type of drone,
                    or False otherwise.
        """
        return self.calculate_forward_velocity(1.0) is not None \
            and self.calculate_right_velocity(1.0) is not None \
            and self.calculate_turn_velocity(1.0) is not None \
            and self.calculate_up_velocity(1.0) is not None

    def update_gimbal_pitch(self, gimbal_pitch: float) -> None:
        """
        Update the pitch of the drone's gimbal (if it has one that can be pitched so as to look up/down).

        .. note::
            If the drone doesn't have a suitable gimbal, this will be a no-op.

        :param gimbal_pitch:   The desired new pitch for the drone's gimbal (in [-1,1], where -1 = down and 1 = up).
        """
        pass
