from smg.rotory.drones.ardrone2 import ARDrone2
from smg.rotory.drones.tello import Tello


class DroneFactory:
    """Used to make drone objects."""

    # PUBLIC STATIC METHODS

    @staticmethod
    def make_drone(drone_type: str):
        """
        Make a drone object of the specified type.

        :param drone_type:  The type of drone object to make.
        :return:            The drone object.
        """
        if drone_type == "ardrone2":
            return ARDrone2()
        elif drone_type == "tello":
            return Tello()
        else:
            raise ValueError(f"Unknown drone type: {drone_type}")
