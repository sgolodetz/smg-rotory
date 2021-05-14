import numpy as np
import threading
import time
import vg

from typing import Callable, Optional, Tuple

from smg.rigging.cameras import SimpleCamera
from smg.rigging.helpers import CameraPoseConverter, CameraUtil

from .drone import Drone


class SimulatedDrone(Drone):
    """An interface that can be used to control a simulated drone."""

    # TYPE ALIASES

    ImageRenderer = Callable[[np.ndarray, Tuple[int, int], Tuple[float, float, float, float]], np.ndarray]

    # NESTED TYPES

    class EState(int):
        pass

    IDLE: EState = EState(0)
    TAKING_OFF: EState = EState(1)
    FLYING: EState = EState(2)
    LANDING: EState = EState(3)

    # CONSTRUCTOR

    def __init__(self, *, image_renderer: ImageRenderer, image_size: Tuple[int, int],
                 intrinsics: Tuple[float, float, float, float]):
        self.__image_renderer: SimulatedDrone.ImageRenderer = image_renderer
        self.__image_size: Tuple[int, int] = image_size
        self.__intrinsics: Tuple[float, float, float, float] = intrinsics
        self.__should_terminate: threading.Event = threading.Event()

        # The simulation variables, together with their locks.
        self.__camera_w_t_c: np.ndarray = np.eye(4)
        self.__drone_w_t_c: np.ndarray = np.eye(4)
        self.__pose_lock: threading.Lock = threading.Lock()

        self.__control_lock: threading.Lock = threading.Lock()
        self.__rc_forward: float = 0.0
        self.__rc_right: float = 0.0
        self.__rc_up: float = 0.0
        self.__rc_yaw: float = 0.0
        self.__state: SimulatedDrone.EState = SimulatedDrone.IDLE

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
        camera_w_t_c, drone_w_t_c = self.__get_poses()
        return self.__image_renderer(camera_w_t_c, self.__image_size, self.__intrinsics)

    def get_image_and_poses(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        camera_w_t_c, drone_w_t_c = self.__get_poses()
        return self.__image_renderer(camera_w_t_c, self.__image_size, self.__intrinsics), camera_w_t_c, drone_w_t_c

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

    def get_state(self) -> EState:
        with self.__control_lock:
            return self.__state

    def land(self) -> None:
        """Tell the drone to land."""
        with self.__control_lock:
            self.__state = SimulatedDrone.LANDING

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
            self.__camera_w_t_c = w_t_c.copy()

    def stop(self) -> None:
        """Tell the drone to stop in mid-air."""
        pass

    def takeoff(self) -> None:
        """Tell the drone to take off."""
        with self.__control_lock:
            self.__state = SimulatedDrone.TAKING_OFF

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

    def __get_poses(self) -> Tuple[np.ndarray, np.ndarray]:
        with self.__pose_lock:
            return self.__camera_w_t_c.copy(), self.__drone_w_t_c.copy()

    def __process_simulation(self) -> None:
        camera_cam: SimpleCamera = CameraUtil.make_default_camera()

        while not self.__should_terminate.is_set():
            with self.__control_lock:
                rc_forward: float = self.__rc_forward
                rc_right: float = self.__rc_right
                rc_up: float = self.__rc_up
                rc_yaw: float = self.__rc_yaw
                state: SimulatedDrone.EState = self.__state

            linear_gain: float = 0.01
            angular_gain: float = 0.01

            camera_cam.move_n(linear_gain * rc_forward)
            camera_cam.move_u(-linear_gain * rc_right)
            camera_cam.rotate(camera_cam.v(), -angular_gain * rc_yaw)

            if state == SimulatedDrone.TAKING_OFF:
                if camera_cam.p()[1] > -1.0:
                    camera_cam.move_v(linear_gain * 0.5)
                else:
                    state = SimulatedDrone.FLYING
            elif state == SimulatedDrone.FLYING:
                camera_cam.move_v(linear_gain * rc_up)
            elif state == SimulatedDrone.LANDING:
                if camera_cam.p()[1] < 0.0:
                    camera_cam.move_v(-linear_gain * 0.5)
                else:
                    state = SimulatedDrone.IDLE

            with self.__control_lock:
                self.__state = state

            drone_cam: SimpleCamera = CameraUtil.make_default_camera()
            drone_cam.set_from(camera_cam)

            if state != SimulatedDrone.IDLE:
                drone_cam.rotate(camera_cam.n(), np.random.normal(0.0, 0.01))
                direction: np.ndarray = np.random.normal(0.0, 1.0, 3)
                length_squared: float = np.dot(direction, direction)
                if length_squared > 0.0:
                    direction = vg.normalize(direction)
                delta: float = np.random.normal(0.0, 0.001)
                drone_cam.move(direction, delta)

            with self.__pose_lock:
                self.__camera_w_t_c = np.linalg.inv(CameraPoseConverter.camera_to_pose(camera_cam))
                self.__drone_w_t_c = np.linalg.inv(CameraPoseConverter.camera_to_pose(drone_cam))

            time.sleep(0.01)
