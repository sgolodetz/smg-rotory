import math
import numpy as np
import threading
import time
import vg

from collections import deque
from typing import Callable, Deque, Optional, Tuple

from smg.rigging.cameras import SimpleCamera
from smg.rigging.helpers import CameraPoseConverter, CameraUtil

from .drone import Drone


class SimulatedDrone(Drone):
    """An interface that can be used to control a simulated drone."""

    # TYPE ALIASES

    # A function that takes a pose, an image size and some intrinsics, and renders an image.
    ImageRenderer = Callable[[np.ndarray, Tuple[int, int], Tuple[float, float, float, float]], np.ndarray]

    # NESTED TYPES

    class EState(int):
        """The states in which a simulated drone can be."""
        pass

    # The drone is on the ground, with its motors switched off.
    IDLE: EState = EState(0)

    # The drone is in the process of performing an automated take-off.
    TAKING_OFF: EState = EState(1)

    # The drone is flying normally.
    FLYING: EState = EState(2)

    # The drone is in the process of performing an automated landing.
    LANDING: EState = EState(3)

    # CONSTRUCTOR

    def __init__(self, *, image_renderer: ImageRenderer, image_size: Tuple[int, int],
                 intrinsics: Tuple[float, float, float, float]):
        """
        Construct a simulated drone.

        :param image_renderer:  TODO
        :param image_size:      TODO
        :param intrinsics:      TODO
        """
        self.__image_renderer: SimulatedDrone.ImageRenderer = image_renderer
        self.__image_size: Tuple[int, int] = image_size
        self.__intrinsics: Tuple[float, float, float, float] = intrinsics
        self.__rc_gimbal_enabled: bool = False
        self.__rc_gimbal_history: Deque[float] = deque()
        self.__should_terminate: threading.Event = threading.Event()

        # The simulation variables, together with their locks.
        self.__camera_w_t_c: np.ndarray = np.eye(4)
        self.__drone_w_t_c: np.ndarray = np.eye(4)
        self.__pose_lock: threading.Lock = threading.Lock()

        self.__control_lock: threading.Lock = threading.Lock()
        self.__rc_forward: float = 0.0
        self.__rc_gimbal: float = 0.0
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
        camera_w_t_c, _ = self.__get_poses()
        return self.__image_renderer(camera_w_t_c, self.__image_size, self.__intrinsics)

    def get_image_and_poses(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        TODO

        :return:    TODO
        """
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
        """
        TODO

        :return:    TODO
        """
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

    def set_gimbal(self, value: float) -> None:
        """
        TODO

        :param value:   TODO
        """
        # TODO
        self.__rc_gimbal_history.append(value)
        if len(self.__rc_gimbal_history) > 10:
            self.__rc_gimbal_history.popleft()

        # TODO:
        avg_value: float = np.mean(self.__rc_gimbal_history)

        # TODO
        if avg_value >= 0.5:
            self.__rc_gimbal_enabled = True

        # TODO
        if self.__rc_gimbal_enabled:
            with self.__control_lock:
                self.__rc_gimbal = 2 * math.pi/2 * (avg_value - 0.5)

    def set_pose(self, w_t_c: np.ndarray) -> None:
        """
        TODO

        :param w_t_c:   TODO
        """
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

            # Wait for the simulation thread to terminate.
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
        """
        TODO

        :return:    TODO
        """
        with self.__pose_lock:
            return self.__camera_w_t_c.copy(), self.__drone_w_t_c.copy()

    def __process_simulation(self) -> None:
        """Run the simulation thread."""
        # TODO
        linear_gain: float = 0.02
        angular_gain: float = 0.02

        # TODO
        camera: SimpleCamera = CameraUtil.make_default_camera()

        # TODO
        while not self.__should_terminate.is_set():
            # TODO
            with self.__control_lock:
                rc_forward: float = self.__rc_forward
                rc_gimbal: float = self.__rc_gimbal
                rc_right: float = self.__rc_right
                rc_up: float = self.__rc_up
                rc_yaw: float = self.__rc_yaw
                state: SimulatedDrone.EState = self.__state

            # TODO
            if state != SimulatedDrone.IDLE:
                camera.move_n(linear_gain * rc_forward)
                camera.move_u(-linear_gain * rc_right)
                camera.rotate(camera.v(), -angular_gain * rc_yaw)

            # TODO
            if state == SimulatedDrone.TAKING_OFF:
                # TODO
                if camera.p()[1] > -1.0:
                    camera.move_v(linear_gain * 0.5)
                else:
                    state = SimulatedDrone.FLYING
            elif state == SimulatedDrone.FLYING:
                # TODO
                camera.move_v(linear_gain * rc_up)
            elif state == SimulatedDrone.LANDING:
                # TODO
                if camera.p()[1] < 0.0:
                    camera.move_v(-linear_gain * 0.5)
                else:
                    state = SimulatedDrone.IDLE

            # TODO
            with self.__control_lock:
                self.__state = state

            # TODO
            camera_cam: SimpleCamera = CameraUtil.make_default_camera()
            camera_cam.set_from(camera)
            camera_cam.rotate(camera.u(), -rc_gimbal)

            # TODO
            drone_cam: SimpleCamera = CameraUtil.make_default_camera()
            drone_cam.set_from(camera)
            if state != SimulatedDrone.IDLE:
                drone_cam.rotate(camera.n(), np.random.normal(0.0, 0.01))
                direction: np.ndarray = np.random.normal(0.0, 1.0, 3)
                length_squared: float = np.dot(direction, direction)
                if length_squared > 0.0:
                    direction = vg.normalize(direction)
                delta: float = np.random.normal(0.0, 0.001)
                drone_cam.move(direction, delta)

            # TODO
            with self.__pose_lock:
                self.__camera_w_t_c = np.linalg.inv(CameraPoseConverter.camera_to_pose(camera_cam))
                self.__drone_w_t_c = np.linalg.inv(CameraPoseConverter.camera_to_pose(drone_cam))

            # TODO
            time.sleep(0.01)
