import math
import os
import sys

os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = "hide"
import pygame

from pprint import pprint
from typing import Dict


class FutabaT6K:
    """An interface that can be used to interact with a Futaba T6K radio controller."""

    # PRIVATE VARIABLES

    """The dead zone threshold. Outputs whose absolute value is smaller than this will be set to zero."""
    __dead_zone_threshold: float = None

    """The joystick device."""
    __device: pygame.joystick.Joystick = None

    """The maximum values observed for each axis during calibration."""
    __maxs: Dict[int, float] = {}

    """The minimum values observed for each axis during calibration."""
    __mins: Dict[int, float] = {}

    # CONSTRUCTORS

    def __init__(self, joystick_idx: int, *, dead_zone_threshold: float = 0.05):
        """
        Construct a FutabaT6K object so as to interact with a Futaba T6K radio controller.

        :param joystick_idx:        The index of the joystick corresponding to the Futaba T6K.
        :param dead_zone_threshold: The dead zone threshold. Outputs whose absolute value is
                                    smaller than this will be set to zero.
        """
        self.__dead_zone_threshold = dead_zone_threshold

        self.__device = pygame.joystick.Joystick(joystick_idx)
        self.__device.init()

        for j in range(self.__device.get_numaxes()):
            self.__mins[j] = float(sys.maxsize)
            self.__maxs[j] = -float(sys.maxsize)

    # SPECIAL METHODS

    def __repr__(self):
        return f"{self.__device.get_name()}, " + \
               repr(dict(zip(self.__mins.keys(), zip(self.__mins.values(), self.__maxs.values())))) + \
               f", Pitch: {self.get_pitch()}, Roll: {self.get_roll()}, Yaw: {self.get_yaw()}"

    def __str__(self):
        return repr(self)

    # PUBLIC METHODS

    def calibrate(self):
        """Calibrate the Futaba T6K."""
        clock = pygame.time.Clock()

        # Allow the user to move the rockers around to explore the range of values available.
        # Stop when Button 0 is set to its "released" state, Button 1 is set to its "pressed" state,
        # and calibration has succeeded.
        while self.__device.get_button(0) != 0 or self.__device.get_button(1) != 1 \
                or self.get_pitch() != 0 or self.get_roll() != 0 or self.get_yaw() != 0:
            # Process and ignore any pygame events.
            for _ in pygame.event.get():
                pass

            # For each axis:
            for j in range(self.__device.get_numaxes()):
                # Use the current axis value to update the minimum and maximum values observed for the axis.
                axis_value = self.__device.get_axis(j)
                if axis_value < self.__mins[j]:
                    self.__mins[j] = axis_value
                if axis_value > self.__maxs[j]:
                    self.__maxs[j] = axis_value

            # Let the user know how the calibration is going.
            pprint(self)

            # Prevent the loop from running at more than 20 Hz (it hurts the user's eyes!).
            clock.tick(20)

    def get_button(self, button_idx: int) -> int:
        """
        Get the value of the specified button on the Futaba T6K.

        :param button_idx:  The button index.
        :return:            The button value.
        """
        return self.__device.get_button(button_idx)

    def get_pitch(self) -> float:
        """
        Get the pitch (forward/backward value) implied by the current axis values.

        :return:    The pitch implied by the current axis values (in [-1,1]).
        """
        return self.__output_value(1, 1, -1)

    def get_roll(self) -> float:
        """
        Get the roll (right/left value) implied by the current axis values.

        :return:    The roll implied by the current axis values (in [-1,1]).
        """
        return self.__output_value(0)

    def get_throttle(self) -> float:
        """
        Get the throttle implied by the current axis values.

        Since the left-hand rocker on my Futaba T6K is a fixed-wing one, using these
        values to control up/down movement would be dangerous, so I don't currently.

        :return:    The throttle implied by the current axis values (in [0,1]).
        """
        return self.__output_value(3, 0, 1)

    def get_yaw(self) -> float:
        """
        Get the yaw implied by the current axis values.

        :return:    The yaw implied by the current axis values (in [-1,1]).
        """
        return self.__output_value(2)

    # PRIVATE METHODS

    def __output_value(self, axis: int, low_output: float = -1.0, high_output: float = 1.0) -> float:
        """
        Compute the value to output for the specified axis on the Futaba T6K.

        :param axis:            The axis.
        :param low_output:      The value to output for the minimum axis value.
        :param high_output:     The value to output for the maximum axis value.
        :return:                The value to output for the specified axis.
        """
        try:
            # Get the raw axis value from the device.
            axis_value = self.__device.get_axis(axis)

            # Normalise the axis value so that it falls in the range [0,1].
            normalised_axis_value = (axis_value - self.__mins[axis]) / (self.__maxs[axis] - self.__mins[axis])

            # Use the normalised axis value to perform a linear blend between the low and high output values.
            output_value = (1 - normalised_axis_value) * low_output + normalised_axis_value * high_output

            # Enforce a dead zone near zero.
            if math.fabs(output_value) < self.__dead_zone_threshold:
                output_value = 0

            return output_value
        except ZeroDivisionError:
            # If the Futaba T6K hasn't been calibrated yet, default to zero.
            return 0.0
