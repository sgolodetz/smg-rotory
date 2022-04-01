import numpy as np
import pygame

from typing import List, Optional, Sequence, Tuple

from .drone_controller import DroneController
from ..drones import Drone


class KeyboardDroneController(DroneController):
    """TODO"""

    # CONSTRUCTOR

    def __init__(self, *, drone: Drone):
        """
        TODO

        :param drone:   TODO
        """
        self.__drone: Drone = drone

    # PUBLIC METHODS

    def iterate(self, *, altitude: Optional[float] = None, events: Optional[List[pygame.event.Event]] = None,
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
        pressed_keys: Sequence[bool] = pygame.key.get_pressed()

        # TODO
        for event in events:
            if event.type == pygame.KEYDOWN:
                # If the user presses the 'u' and 'left shift' keys, take off.
                if event.key == pygame.K_u and pressed_keys[pygame.K_LSHIFT]:
                    self.__drone.takeoff()

                # If the user presses the 'o' and 'left shift' keys, land.
                elif event.key == pygame.K_o and pressed_keys[pygame.K_LSHIFT]:
                    self.__drone.land()

        # TODO
        if pressed_keys[pygame.K_i]:
            self.__drone.move_forward(0.5)
        elif pressed_keys[pygame.K_k]:
            self.__drone.move_forward(-0.5)
        else:
            self.__drone.move_forward(0.0)

        # TODO
        if pressed_keys[pygame.K_j] and pressed_keys[pygame.K_LSHIFT]:
            self.__drone.turn(-1.0)
        elif pressed_keys[pygame.K_l] and pressed_keys[pygame.K_LSHIFT]:
            self.__drone.turn(1.0)
        else:
            self.__drone.turn(0.0)

        # TODO
        if pressed_keys[pygame.K_j] and not pressed_keys[pygame.K_LSHIFT]:
            self.__drone.move_right(-0.5)
        elif pressed_keys[pygame.K_l] and not pressed_keys[pygame.K_LSHIFT]:
            self.__drone.move_right(0.5)
        else:
            self.__drone.move_right(0.0)

        # TODO
        if pressed_keys[pygame.K_u] and not pressed_keys[pygame.K_LSHIFT]:
            self.__drone.move_up(0.5)
        elif pressed_keys[pygame.K_o] and not pressed_keys[pygame.K_LSHIFT]:
            self.__drone.move_up(-0.5)
        else:
            self.__drone.move_up(0.0)
