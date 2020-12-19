from .drones.ardrone2 import ARDrone2
from .drones.drone import Drone
from .drones.tello import Tello


class DroneFactory:
    """Used to make drone objects."""

    # PUBLIC STATIC METHODS

    @staticmethod
    def make_drone(drone_type: str, **kwargs) -> Drone:
        """
        Make a drone object of the specified type.

        :param drone_type:  The type of drone object to make.
        :param kwargs:      Any keyword arguments to pass to the drone constructor.
        :return:            The drone object.
        """
        if drone_type == "ardrone2":
            return ARDrone2(**kwargs)
        elif drone_type == "tello":
            return Tello(**kwargs)
        else:
            raise ValueError(f"Unknown drone type: {drone_type}")
