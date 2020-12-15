import numpy as np

from typing import Optional


class MonocularPoseCorrector:
    """TODO"""
    # FIXME: This should ultimately be moved elsewhere.

    # CONSTRUCTOR

    def __init__(self, *, debug: bool = True):
        """
        TODO

        :param debug:   TODO
        """
        self.__debug: bool = debug
        self.__reference_relocaliser_w_t_c: Optional[np.ndarray] = None
        self.__reference_tracker_i_t_c: Optional[np.ndarray] = None
        self.__scale: float = 1.0
        self.__scale_count: int = 0
        self.__scale_sum: float = 0.0

    # PUBLIC METHODS

    def apply(self, tracker_i_t_c: np.ndarray) -> np.ndarray:
        """
        TODO

        :param tracker_i_t_c:   TODO
        :return:                TODO
        """
        scaled_reference_tracker_i_t_c: np.ndarray = self.__reference_tracker_i_t_c.copy()
        scaled_reference_tracker_i_t_c[0:3, 3] *= self.__scale
        scaled_tracker_i_t_c: np.ndarray = tracker_i_t_c.copy()
        scaled_tracker_i_t_c[0:3, 3] *= self.__scale
        # # wTc = wTi . iTc
        return self.__reference_relocaliser_w_t_c @ np.linalg.inv(scaled_reference_tracker_i_t_c) @ scaled_tracker_i_t_c

    def calibrate(self, tracker_i_t_c: np.ndarray, relocaliser_w_t_c: np.ndarray, *, min_norm: float = 0.1) -> None:
        """
        TODO

        :param tracker_i_t_c:       TODO
        :param relocaliser_w_t_c:   TODO
        :param min_norm:            TODO
        """
        tracker_offset: np.ndarray = tracker_i_t_c[0:3, 3] - self.__reference_tracker_i_t_c[0:3, 3]
        relocaliser_offset: np.ndarray = relocaliser_w_t_c[0:3, 3] - self.__reference_relocaliser_w_t_c[0:3, 3]
        tracker_norm: float = np.linalg.norm(tracker_offset)
        relocaliser_norm: float = np.linalg.norm(relocaliser_offset)
        if tracker_norm > 0 and relocaliser_norm >= min_norm:
            scale_estimate: float = relocaliser_norm / tracker_norm
            self.__scale_sum += scale_estimate
            self.__scale_count += 1
            self.__scale = self.__scale_sum / self.__scale_count
            if self.__debug:
                print(relocaliser_norm, tracker_norm * self.__scale, scale_estimate, self.__scale)

    def has_reference(self) -> bool:
        """
        TODO

        :return:    TODO
        """
        return self.__reference_relocaliser_w_t_c is not None

    def maintain_height(self) -> None:
        """
        TODO
        """
        # TODO
        pass

    def reset(self) -> None:
        """
        Reset the pose corrector.
        """
        self.__reference_relocaliser_w_t_c = None
        self.__reference_tracker_i_t_c = None
        self.__scale = 1.0
        self.__scale_count = 0
        self.__scale_sum = 0.0

    def set_reference(self, tracker_i_t_c: np.ndarray, relocaliser_w_t_c: np.ndarray) -> None:
        """
        TODO

        :param tracker_i_t_c:       TODO
        :param relocaliser_w_t_c:   TODO
        """
        self.__reference_relocaliser_w_t_c = relocaliser_w_t_c
        self.__reference_tracker_i_t_c = tracker_i_t_c
        self.__scale = 1.0
        self.__scale_count = 0
        self.__scale_sum = 0.0
