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
                 intrinsics: Tuple[float, float, float, float],
                 linear_gain: float = 0.02, angular_gain: float = 0.02):
        """
        Construct a simulated drone.

        :param image_renderer:  A function that can be used to render a synthetic image of what the drone can see
                                from the current pose of its camera.
        :param image_size:      The size of the synthetic images that should be rendered for the drone, as a
                                (width, height) tuple.
        :param intrinsics:      The camera intrinsics to use when rendering the synthetic images for the drone,
                                as an (fx, fy, cx, cy) tuple.
        :param linear_gain:     TODO
        :param angular_gain:    TODO
        """
        self.__gimbal_input_history: Deque[float] = deque()
        self.__image_renderer: SimulatedDrone.ImageRenderer = image_renderer
        self.__image_size: Tuple[int, int] = image_size
        self.__intrinsics: Tuple[float, float, float, float] = intrinsics
        self.__linear_gain: float = linear_gain
        self.__angular_gain: float = angular_gain
        self.__should_terminate: threading.Event = threading.Event()

        # The simulation variables, together with their locks.
        self.__gimbal_pitch: float = 0.0
        self.__rc_forward: float = 0.0
        self.__rc_right: float = 0.0
        self.__rc_up: float = 0.0
        self.__rc_yaw: float = 0.0
        self.__state: SimulatedDrone.EState = SimulatedDrone.IDLE
        self.__input_lock: threading.Lock = threading.Lock()

        self.__camera_w_t_c: np.ndarray = np.eye(4)
        self.__chassis_w_t_c: np.ndarray = np.eye(4)
        self.__output_lock: threading.Lock = threading.Lock()

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
        Get the most recent image received from the drone, together with the poses of the drone's camera and chassis.

        .. note::
            In our simulation, the drone's camera and chassis have separate poses to allow the drone to wobble around
            in the air and thereby make the simulation look a bit more realistic.

        :return:    The most recent image received from the drone, together with the poses of the drone's camera
                    and chassis, as an (image, camera pose, chassis pose) tuple.
        """
        camera_w_t_c, chassis_w_t_c = self.__get_poses()
        return self.__image_renderer(camera_w_t_c, self.__image_size, self.__intrinsics), camera_w_t_c, chassis_w_t_c

    def get_image_size(self) -> Tuple[int, int]:
        """
        Get the size of the images captured by the drone.

        :return:    The size of the images captured by the drone, as a (width, height) tuple.
        """
        return self.__image_size

    def get_intrinsics(self) -> Optional[Tuple[float, float, float, float]]:
        """
        Get the camera intrinsics.

        :return:    The camera intrinsics as an (fx, fy, cx, cy) tuple.
        """
        return self.__intrinsics

    def get_state(self) -> EState:
        """
        Get the current state of the drone.

        :return:    The current state of the drone.
        """
        with self.__input_lock:
            return self.__state

    def land(self) -> None:
        """Tell the drone to land."""
        with self.__input_lock:
            self.__state = SimulatedDrone.LANDING

    def move_forward(self, rate: float) -> None:
        """
        Tell the drone to move forward at the specified rate.

        .. note::
            This can also be used to move backwards (by specifying a negative rate).

        :param rate:     The rate at which the drone should move forward (in [-1,1]).
        """
        with self.__input_lock:
            self.__rc_forward = rate

    def move_right(self, rate: float) -> None:
        """
        Tell the drone to move to the right at the specified rate.

        .. note::
            This can also be used to move to the left (by specifying a negative rate).

        :param rate:    The rate at which the drone should move to the right (in [-1,1]).
        """
        with self.__input_lock:
            self.__rc_right = rate

    def move_up(self, rate: float) -> None:
        """
        Tell the drone to move up at the specified rate.

        .. note::
            This can also be used to move down (by specifying a negative rate).

        :param rate:    The rate at which the drone should move up (in [-1,1]).
        """
        with self.__input_lock:
            self.__rc_up = rate

    def stop(self) -> None:
        """Tell the drone to stop in mid-air."""
        with self.__input_lock:
            self.__rc_forward = 0
            self.__rc_right = 0
            self.__rc_up = 0
            self.__rc_yaw = 0

    def takeoff(self) -> None:
        """Tell the drone to take off."""
        with self.__input_lock:
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
        with self.__input_lock:
            self.__rc_yaw = rate

    def update_gimbal_pitch(self, gimbal_input: float) -> None:
        """
        Update the pitch of the drone's gimbal.

        .. note::
            We assume the drone has a gimbal that can be pitched so as to look up/down.
        .. note::
            This function doesn't update the pitch of the drone's gimbal directly, since we don't want jerky gimbal
            input values to make the view oscillate up and down. Instead, it adds gimbal input values to a history
            of max size N. The pitch of the gimbal is then set to be the mean of the most recent <= N values added
            to this history, which allows it to be changed smoothly over time, albeit with a bit of lag.

        :param gimbal_input:   The value to add to the gimbal history (in [-1,1], where -1 = down and 1 = up).
        """
        # Add the value to the gimbal history. If that makes the gimbal history too big, discard the oldest value.
        self.__gimbal_input_history.append(gimbal_input)
        if len(self.__gimbal_input_history) > 10:
            self.__gimbal_input_history.popleft()

        # Compute and set the new gimbal pitch.
        gimbal_pitch: float = np.mean(self.__gimbal_input_history) * math.pi / 2
        with self.__input_lock:
            self.__gimbal_pitch = gimbal_pitch

    # PRIVATE METHODS

    def __get_poses(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get the poses of the drone's camera and chassis.

        :return:    The poses of the drone's camera and chassis, as a (camera pose, chassis pose) tuple.
        """
        with self.__output_lock:
            return self.__camera_w_t_c.copy(), self.__chassis_w_t_c.copy()

    def __process_simulation(self) -> None:
        """Run the simulation thread."""
        # Construct the camera corresponding to the master pose for the drone (the poses of its camera and chassis
        # will be derived from this each frame).
        master_cam: SimpleCamera = CameraUtil.make_default_camera()

        # Until the simulation should terminate:
        while not self.__should_terminate.is_set():
            # Make a copy of the inputs so that we only need to hold the lock very briefly.
            with self.__input_lock:
                gimbal_pitch: float = self.__gimbal_pitch
                rc_forward: float = self.__rc_forward
                rc_right: float = self.__rc_right
                rc_up: float = self.__rc_up
                rc_yaw: float = self.__rc_yaw
                state: SimulatedDrone.EState = self.__state

            # Provided the drone's not stationary on the ground, process any horizontal movements that are requested.
            if state != SimulatedDrone.IDLE:
                master_cam.move_n(self.__linear_gain * rc_forward)
                master_cam.move_u(-self.__linear_gain * rc_right)
                master_cam.rotate(master_cam.v(), -self.__angular_gain * rc_yaw)

            # Depending on the drone's state:
            if state == SimulatedDrone.TAKING_OFF:
                # If the drone's taking off, move it upwards at a constant rate until it's 1m off the ground,
                # then switch to the flying state. (Note that y points downwards in our coordinate system!)
                if master_cam.p()[1] > -1.0:
                    master_cam.move_v(self.__linear_gain * 0.5)
                else:
                    state = SimulatedDrone.FLYING
            elif state == SimulatedDrone.FLYING:
                # If the drone's flying, process any vertical movements that are requested.
                master_cam.move_v(self.__linear_gain * rc_up)
            elif state == SimulatedDrone.LANDING:
                # If the drone's landing, move it downwards at a constant rate until it's on the ground,
                # then switch to the idle state. (Note that y points downwards in our coordinate system!)
                if master_cam.p()[1] < 0.0:
                    master_cam.move_v(-self.__linear_gain * 0.5)
                else:
                    state = SimulatedDrone.IDLE

            # Update the global version of the state to actually effect the state change.
            with self.__input_lock:
                self.__state = state

            # Construct a camera corresponding to the pose of the drone's camera.
            camera_cam: SimpleCamera = CameraUtil.make_default_camera()
            camera_cam.set_from(master_cam)
            camera_cam.rotate(master_cam.u(), -gimbal_pitch)

            # Construct a camera corresponding to the pose of the drone's chassis. Provided we're not in the idle
            # state, this is derived by adding some Gaussian noise to the master pose.
            chassis_cam: SimpleCamera = CameraUtil.make_default_camera()
            chassis_cam.set_from(master_cam)
            if state != SimulatedDrone.IDLE:
                chassis_cam.rotate(master_cam.n(), np.random.normal(0.0, 0.01))
                direction: np.ndarray = np.random.normal(0.0, 1.0, 3)
                length_squared: float = np.dot(direction, direction)
                if length_squared > 0.0:
                    direction = vg.normalize(direction)
                delta: float = np.random.normal(0.0, 0.001)
                chassis_cam.move(direction, delta)

            # Update the the poses of the drone's camera and chassis.
            with self.__output_lock:
                self.__camera_w_t_c = np.linalg.inv(CameraPoseConverter.camera_to_pose(camera_cam))
                self.__chassis_w_t_c = np.linalg.inv(CameraPoseConverter.camera_to_pose(chassis_cam))

            # Wait momentarily before processing the next iteration of the simulation.
            time.sleep(0.01)
