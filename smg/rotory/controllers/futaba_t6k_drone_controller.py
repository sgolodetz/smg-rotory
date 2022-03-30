import numpy as np
import pygame

from typing import List, Optional, Tuple

from smg.joysticks import FutabaT6K

from .drone_controller import DroneController
from ..drones import Drone


class FutabaT6KDroneController(DroneController):
    """TODO"""

    # CONSTRUCTOR

    def __init__(self, drone: Drone, joystick: FutabaT6K):
        """
        TODO

        :param drone:       TODO
        :param joystick:    TODO
        """
        self.__drone: Drone = drone
        self.__joystick: FutabaT6K = joystick

    # PUBLIC METHODS

    def update(self, *, altitude: Optional[float] = None, events: Optional[List[pygame.event.Event]] = None,
               image: np.ndarray, image_timestamp: Optional[float] = None,
               intrinsics: Tuple[float, float, float, float], tracker_c_t_i: Optional[np.ndarray] = None) -> None:
        """
        Run an iteration of the controller.

        :param altitude:            The most recent altitude (in m) for the drone, as measured by any height sensor
                                    it is carrying (optional).
        :param events:              An optional list of PyGame events that have happened since the last update.
        :param image:               The most recent image from the drone.
        :param image_timestamp:     The timestamp of the most recent image from the drone (optional).
        :param intrinsics:          The intrinsics of the drone's camera.
        :param tracker_c_t_i:       A transformation from initial camera space to current camera space, as estimated
                                    by any tracker that's running (optional). Note that if the tracker is a monocular
                                    one, the transformation will be non-metric.
        """
        # TODO
        if events is None:
            events = []

        # TODO
        for event in events:
            if event.type == pygame.JOYBUTTONDOWN:
                # If Button 0 on the Futaba T6K is set to its "pressed" state, take off.
                if event.button == 0:
                    self.__drone.takeoff()
            elif event.type == pygame.JOYBUTTONUP:
                # If Button 0 on the Futaba T6K is set to its "released" state, land.
                if event.button == 0:
                    self.__drone.land()

        # Update the movement of the drone based on the pitch, roll and yaw values output by the Futaba T6K.
        self.__drone.move_forward(self.__joystick.get_pitch())
        self.__drone.turn(self.__joystick.get_yaw())

        if self.__joystick.get_button(1) == 0:
            self.__drone.move_right(0)
            self.__drone.move_up(self.__joystick.get_roll())
        else:
            self.__drone.move_right(self.__joystick.get_roll())
            self.__drone.move_up(0)
