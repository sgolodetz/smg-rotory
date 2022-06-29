import numpy as np


class Beacon:
    """A beacon (a transmitter that can provide range measurements for use in localisation)."""

    # NESTED TYPES

    class EBeaconType(int):
        """The different types of beacon that are possible."""
        pass

    # A fake beacon with a known position, manually placed in the scene by the user.
    BT_FAKE = EBeaconType(0)
    # A localised beacon (one whose position has been estimated based on range measurements).
    BT_LOCALISED = EBeaconType(1)

    # CONSTRUCTOR

    def __init__(self, position: np.ndarray, max_range: float, beacon_type: EBeaconType):
        """
        Construct a beacon.

        :param position:    The position of the beacon.
        :param max_range:   The maximum range of the beacon (in m). This is the maximum distance at which it is able
                            to provide range measurements.
        :param beacon_type: The type of the beacon.
        """
        self.__beacon_type: Beacon.EBeaconType = beacon_type
        self.__max_range: float = max_range
        self.__position: np.ndarray = position

    # PROPERTIES

    @property
    def beacon_type(self) -> EBeaconType:
        """
        Get the type of the beacon.

        :return:    The type of the beacon.
        """
        return self.__beacon_type

    @property
    def max_range(self) -> float:
        """
        Get the maximum range of the beacon (in m).

        .. note::
            This is the maximum distance at which it is able to provide range measurements.

        :return:    The maximum range of the beacon (in m).
        """
        return self.__max_range

    @property
    def position(self) -> np.ndarray:
        """
        Get the position of the beacon.

        :return:    The position of the beacon.
        """
        return self.__position
