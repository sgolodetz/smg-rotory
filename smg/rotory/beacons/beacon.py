import numpy as np


class Beacon:
    """TODO"""

    # NESTED TYPES

    class EBeaconType(int):
        """The different types of beacon that are possible."""
        pass

    # TODO
    BT_FAKE = EBeaconType(0)
    # TODO
    BT_LOCALISED = EBeaconType(1)

    # CONSTRUCTOR

    def __init__(self, position: np.ndarray, max_range: float, beacon_type: EBeaconType):
        """
        TODO

        :param position:    TODO
        :param max_range:   TODO
        :param beacon_type: TODO
        """
        self.__beacon_type: Beacon.EBeaconType = beacon_type
        self.__max_range: float = max_range
        self.__position: np.ndarray = position

    # PROPERTIES

    @property
    def beacon_type(self) -> EBeaconType:
        """
        TODO

        :return:    TODO
        """
        return self.__beacon_type

    @property
    def max_range(self) -> float:
        """
        TODO

        :return:    TODO
        """
        return self.__max_range

    @property
    def position(self) -> np.ndarray:
        """
        TODO

        :return:    TODO
        """
        return self.__position
