import numpy as np
import threading

from typing import Callable, Optional, Tuple

from .drone import Drone


class SimulatedDrone(Drone):
    """An interface that can be used to control a simulated drone."""

    # CONSTRUCTOR

    def __init__(self, *, image_renderer: Callable[[np.ndarray], np.ndarray], image_size: Tuple[int, int],
                 intrinsics: Tuple[float, float, float, float]):
        self.__image_renderer: Callable[[np.ndarray], np.ndarray] = image_renderer
        self.__image_size: Tuple[int, int] = image_size
        self.__intrinsics: Tuple[float, float, float, float] = intrinsics
        self.__should_terminate: threading.Event = threading.Event()

        # The simulation variables, together with their lock.
        self.__rc_forward: float = 0.0
        self.__rc_right: float = 0.0
        self.__rc_up: float = 0.0
        self.__rc_yaw: float = 0.0
        self.__simulation_lock: threading.Lock = threading.Lock()
        self.__w_t_c: np.ndarray = np.eye(4)

        # Start the simulation.
        self.__simulation_thread: threading.Thread = threading.Thread(target=self.__process_simulation)
        self.__simulation_thread.start()

    # SPECIAL METHODS

    def __enter__(self):
        """No-op (needed to allow the drone object's lifetime to be managed by a with statement)."""
        return self

    def __exit__(self, exception_type, exception_value, traceback):
        """Destroy the drone object at the end of the with statement that's used to manage its lifetime."""
        self.terminate()

    # PUBLIC METHODS

    def get_battery_level(self) -> Optional[int]:
        """
        Try to get the most recently received value of the remaining battery %.

        :return:    The most recently received value of the remaining battery %, if available, or None otherwise.
        """
        return 100

    def get_image(self) -> np.ndarray:
        """
        Get the most recent image received from the drone.

        :return:    The most recent image received from the drone.
        """
        return self.__image_renderer(self.get_pose())

    def get_image_and_pose(self) -> Tuple[np.ndarray, np.ndarray]:
        w_t_c: np.ndarray = self.get_pose()
        return self.__image_renderer(w_t_c), w_t_c

    def get_image_size(self) -> Tuple[int, int]:
        """
        Get the size of the images captured by the drone.

        :return:    The size of the images captured by the drone, as a (width, height) tuple.
        """
        return self.__image_size

    def get_intrinsics(self) -> Optional[Tuple[float, float, float, float]]:
        """
        Get the camera intrinsics, if known.

        :return:    The camera intrinsics as an (fx, fy, cx, cy) tuple, if known, or None otherwise.
        """
        return self.__intrinsics

    def get_pose(self) -> np.ndarray:
        with self.__simulation_lock:
            return self.__w_t_c.copy()

    def land(self) -> None:
        """Tell the drone to land."""
        pass

    def move_forward(self, rate: float) -> None:
        """
        Tell the drone to move forward at the specified rate.

        .. note::
            This can also be used to move backwards (by specifying a negative rate).

        :param rate:     The rate at which the drone should move forward (in [-1,1]).
        """
        pass

    def move_right(self, rate: float) -> None:
        """
        Tell the drone to move to the right at the specified rate.

        .. note::
            This can also be used to move to the left (by specifying a negative rate).

        :param rate:    The rate at which the drone should move to the right (in [-1,1]).
        """
        pass

    def move_up(self, rate: float) -> None:
        """
        Tell the drone to move up at the specified rate.

        .. note::
            This can also be used to move down (by specifying a negative rate).

        :param rate:    The rate at which the drone should move up (in [-1,1]).
        """
        pass

    def set_pose(self, w_t_c: np.ndarray) -> None:
        with self.__simulation_lock:
            self.__w_t_c = w_t_c.copy()

    def stop(self) -> None:
        """Tell the drone to stop in mid-air."""
        pass

    def takeoff(self) -> None:
        """Tell the drone to take off."""
        pass

    def terminate(self) -> None:
        """Tell the drone to terminate."""
        if not self.__should_terminate.is_set():
            self.__should_terminate.set()

            # Wait for all of the threads to terminate.
            self.__simulation_thread.join()

    def turn(self, rate: float) -> None:
        """
        Tell the drone to turn at the specified rate.

        :param rate:    The rate at which the drone should turn (in [-1,1]).
        """
        pass

    # PRIVATE METHODS

    def __process_simulation(self) -> None:
        while not self.__should_terminate.is_set():
            with self.__simulation_lock:
                pass
