import numpy as np


class Beacon:
    """TODO"""

    # CONSTRUCTOR

    def __init__(self, position: np.ndarray, max_range: float):
        """
        TODO

        :param position:    TODO
        :param max_range:   TODO
        """
        self.__max_range: float = max_range
        self.__position: np.ndarray = position

    # PROPERTIES

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
