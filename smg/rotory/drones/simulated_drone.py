import numpy as np
import threading
import time

from typing import Callable, Optional, Tuple

from smg.rigging.cameras import SimpleCamera
from smg.rigging.helpers import CameraPoseConverter, CameraUtil

from .drone import Drone


class SimulatedDrone(Drone):
    """An interface that can be used to control a simulated drone."""

    # TYPE ALIASES

    ImageRenderer = Callable[[np.ndarray, Tuple[int, int], Tuple[float, float, float, float]], np.ndarray]

    # CONSTRUCTOR

    def __init__(self, *, image_renderer: ImageRenderer, image_size: Tuple[int, int],
                 intrinsics: Tuple[float, float, float, float]):
        self.__image_renderer: SimulatedDrone.ImageRenderer = image_renderer
        self.__image_size: Tuple[int, int] = image_size
        self.__intrinsics: Tuple[float, float, float, float] = intrinsics
        self.__should_terminate: threading.Event = threading.Event()

        # The simulation variables, together with their locks.
        self.__rc_forward: float = 0.0
        self.__rc_right: float = 0.0
        self.__rc_up: float = 0.0
        self.__rc_yaw: float = 0.0
        self.__control_lock: threading.Lock = threading.Lock()
        self.__pose_lock: threading.Lock = threading.Lock()
        # self.__simulation_lock: threading.Lock = threading.Lock()
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
        return self.__image_renderer(self.get_pose(), self.__image_size, self.__intrinsics)

    def get_image_and_pose(self) -> Tuple[np.ndarray, np.ndarray]:
        w_t_c: np.ndarray = self.get_pose()
        return self.__image_renderer(w_t_c, self.__image_size, self.__intrinsics), w_t_c

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
        with self.__pose_lock:
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
        with self.__control_lock:
            self.__rc_forward = rate

    def move_right(self, rate: float) -> None:
        """
        Tell the drone to move to the right at the specified rate.

        .. note::
            This can also be used to move to the left (by specifying a negative rate).

        :param rate:    The rate at which the drone should move to the right (in [-1,1]).
        """
        with self.__control_lock:
            self.__rc_right = rate

    def move_up(self, rate: float) -> None:
        """
        Tell the drone to move up at the specified rate.

        .. note::
            This can also be used to move down (by specifying a negative rate).

        :param rate:    The rate at which the drone should move up (in [-1,1]).
        """
        with self.__control_lock:
            self.__rc_up = rate

    def set_pose(self, w_t_c: np.ndarray) -> None:
        with self.__pose_lock:
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
        with self.__control_lock:
            self.__rc_yaw = rate

    # PRIVATE METHODS

    def __process_simulation(self) -> None:
        camera: SimpleCamera = CameraUtil.make_default_camera()

        while not self.__should_terminate.is_set():
            with self.__control_lock:
                rc_forward: float = self.__rc_forward
                rc_right: float = self.__rc_right
                rc_up: float = self.__rc_up
                rc_yaw: float = self.__rc_yaw

            # rc_forward = -1.0
            # rc_up = 1.0
            # rc_yaw = -1.0

            linear_gain: float = 0.01
            angular_gain: float = 0.01
            camera.move_n(linear_gain * rc_forward)
            camera.move_u(-linear_gain * rc_right)
            camera.move_v(linear_gain * rc_up)
            camera.rotate(camera.v(), -angular_gain * rc_yaw)

            with self.__pose_lock:
                self.__w_t_c = np.linalg.inv(CameraPoseConverter.camera_to_pose(camera))

            time.sleep(0.01)
