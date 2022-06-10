import math
import numpy as np
import threading
import time
import vg

from collections import deque
from timeit import default_timer as timer
from typing import Callable, Deque, Optional, Tuple

from smg.rigging.cameras import Camera, SimpleCamera
from smg.rigging.helpers import CameraPoseConverter, CameraUtil

from .drone import Drone


class SimulatedDrone(Drone):
    """An interface that can be used to control a simulated drone."""

    # TYPE ALIASES

    # A function that takes the drone's camera and chassis poses (in that order), an image size and some intrinsics,
    # and renders an image.
    ImageRenderer = Callable[[np.ndarray, np.ndarray, Tuple[int, int], Tuple[float, float, float, float]], np.ndarray]

    # A function that can be called iteratively to try to make the drone land. It returns the drone state after
    # each call. If the drone is still flying, the landing has failed. If the drone's landing, we need to continue
    # calling the function iteratively. If the drone has landed, we can stop.
    LandingController = Callable[[SimpleCamera, float], Drone.EState]

    # A function that can be called iteratively to try to make the drone take off. It returns the drone state after
    # each call. If the drone is still on the ground, the takeoff has failed. If the drone's taking off, we need to
    # continue calling the function iteratively. If the drone is flying, we can stop.
    TakeoffController = Callable[[SimpleCamera, float], Drone.EState]

    # CONSTRUCTOR

    def __init__(self, *, drone_origin: Optional[SimpleCamera] = None,
                 image_renderer: Optional[ImageRenderer] = None, image_size: Tuple[int, int] = (640, 480),
                 intrinsics: Tuple[float, float, float, float] = (500, 500, 320, 240)):
        """
        Construct a simulated drone.

        .. note::
            If drone_origin is set to None, the initial origin for the drone will be the world-space origin.

        :param drone_origin:    The initial origin for the drone (optional).
        :param image_renderer:  An optional function that can be used to render a synthetic image of what the drone
                                can see from the current pose of its camera.
        :param image_size:      The size of the synthetic images that should be rendered for the drone, as a
                                (width, height) tuple.
        :param intrinsics:      The camera intrinsics to use when rendering the synthetic images for the drone,
                                as an (fx, fy, cx, cy) tuple.
        """
        self.__gimbal_input_history: Deque[float] = deque()
        self.__image_renderer: SimulatedDrone.ImageRenderer = image_renderer \
            if image_renderer is not None else SimulatedDrone.blank_image_renderer
        self.__image_size: Tuple[int, int] = image_size
        self.__intrinsics: Tuple[float, float, float, float] = intrinsics
        self.__landing_controller: Optional[SimulatedDrone.LandingController] = self.default_landing_controller
        self.__should_terminate: threading.Event = threading.Event()
        self.__takeoff_controller: Optional[SimulatedDrone.TakeoffController] = self.default_takeoff_controller

        # The simulation variables, together with their locks.
        self.__drone_origin: SimpleCamera = CameraUtil.make_default_camera()
        if drone_origin is not None:
            self.__drone_origin.set_from(drone_origin)
        self.__drone_origin_changed: threading.Event = threading.Event()
        self.__gimbal_pitch: float = 0.0
        self.__rc_forward: float = 0.0
        self.__rc_right: float = 0.0
        self.__rc_up: float = 0.0
        self.__rc_yaw: float = 0.0
        self.__state: Drone.EState = Drone.IDLE
        self.__input_lock: threading.Lock = threading.Lock()

        self.__camera_w_t_c: np.ndarray = np.linalg.inv(CameraPoseConverter.camera_to_pose(self.__drone_origin))
        self.__chassis_w_t_c: np.ndarray = self.__camera_w_t_c.copy()
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

    # PUBLIC STATIC METHODS

    @staticmethod
    def blank_image_renderer(camera_w_t_c: np.ndarray, chassis_w_t_c, image_size: Tuple[int, int],
                             intrinsics: Tuple[float, float, float, float]) -> np.ndarray:
        """
        Render a blank image as a dummy version of what the drone can see of the scene from its current pose.

        .. note::
            This is useful when we want to use a simulated drone but don't need to worry about what it can see.

        :param camera_w_t_c:    The pose of the drone's camera.
        :param chassis_w_t_c:   The pose of the drone's chassis.
        :param image_size:      The size of image to render.
        :param intrinsics:      The camera intrinsics.
        :return:                The rendered image.
        """
        width, height = image_size
        return np.zeros((height, width, 3), dtype=np.uint8)

    # PUBLIC METHODS

    def calculate_forward_rate(self, m_per_s: float, *, allow_clipping: bool = True) -> Optional[float]:
        """
        TODO

        :param m_per_s:         TODO
        :param allow_clipping:  TODO
        :return:                TODO
        """
        return SimulatedDrone.__calculate_rate(
            units_per_s=m_per_s,
            max_units_per_s=abs(self.calculate_forward_velocity(rate=1.0)),
            allow_clipping=allow_clipping
        )

    def calculate_forward_velocity(self, rate: float) -> Optional[float]:
        """
        TODO

        :param rate:    TODO
        :return:        TODO
        """
        return rate * 2.0

    def calculate_right_rate(self, m_per_s: float, *, allow_clipping: bool = True) -> Optional[float]:
        """
        TODO

        :param m_per_s:         TODO
        :param allow_clipping:  TODO
        :return:                TODO
        """
        return SimulatedDrone.__calculate_rate(
            units_per_s=m_per_s,
            max_units_per_s=abs(self.calculate_right_velocity(rate=1.0)),
            allow_clipping=allow_clipping
        )

    def calculate_right_velocity(self, rate: float) -> Optional[float]:
        """
        TODO

        :param rate:    TODO
        :return:        TODO
        """
        return rate * 2.0

    def calculate_turn_rate(self, rad_per_s: float, *, allow_clipping: bool = True) -> Optional[float]:
        """
        TODO

        :param rad_per_s:       TODO
        :param allow_clipping:  TODO
        :return:                TODO
        """
        # TODO
        return SimulatedDrone.__calculate_rate(
            units_per_s=rad_per_s,
            max_units_per_s=abs(self.calculate_turn_velocity(rate=1.0)),
            allow_clipping=allow_clipping
        )

    def calculate_turn_velocity(self, rate: float) -> Optional[float]:
        """
        TODO

        :param rate:    TODO
        :return:        TODO
        """
        return -rate * np.pi / 2

    def calculate_up_rate(self, m_per_s: float, *, allow_clipping: bool = True) -> Optional[float]:
        """
        TODO

        :param m_per_s:         TODO
        :param allow_clipping:  TODO
        :return:                TODO
        """
        return SimulatedDrone.__calculate_rate(
            units_per_s=m_per_s,
            max_units_per_s=abs(self.calculate_up_velocity(rate=1.0)),
            allow_clipping=allow_clipping
        )

    def calculate_up_velocity(self, rate: float) -> Optional[float]:
        """
        TODO

        :param rate:    TODO
        :return:        TODO
        """
        return rate * 2.0

    def default_landing_controller(self, drone_cur: SimpleCamera, time_offset: float) -> Drone.EState:
        """
        Run an iteration of the default landing controller.

        :param drone_cur:   A camera corresponding to the drone's current pose.
        :param time_offset: TODO
        :return:            The state of the drone after this iteration of the controller.
        """
        # Make a local copy of the drone's origin.
        with self.__input_lock:
            drone_origin: SimpleCamera = CameraUtil.make_default_camera()
            drone_origin.set_from(self.__drone_origin)

        # Move the drone downwards at a constant rate until it's no higher than the drone's origin,
        # then switch to the idle state. (Note that y points downwards in our coordinate system!)
        if drone_cur.p()[1] < drone_origin.p()[1]:
            target_velocity: float = -1.0
            velocity: float = self.clip_up_velocity(target_velocity)
            drone_cur.move_v(time_offset * velocity)
            return Drone.LANDING
        else:
            return Drone.IDLE

    def default_takeoff_controller(self, drone_cur: SimpleCamera, time_offset: float) -> Drone.EState:
        """
        Run an iteration of the default takeoff controller.

        :param drone_cur:   A camera corresponding to the drone's current pose.
        :param time_offset: TODO
        :return:            The state of the drone after this iteration of the controller.
        """
        # Make a local copy of the drone's origin.
        with self.__input_lock:
            drone_origin: SimpleCamera = CameraUtil.make_default_camera()
            drone_origin.set_from(self.__drone_origin)

        # Move the drone upwards at a constant rate until it's at least 1m above the drone's origin,
        # then switch to the flying state. (Note that y points downwards in our coordinate system!)
        if drone_cur.p()[1] > drone_origin.p()[1] - 1.0:
            target_velocity: float = 1.0
            velocity: float = self.clip_up_velocity(target_velocity)
            drone_cur.move_v(time_offset * velocity)
            return Drone.TAKING_OFF
        else:
            return Drone.FLYING

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
        camera_w_t_c, chassis_w_t_c = self.__get_poses()
        return self.__image_renderer(camera_w_t_c, chassis_w_t_c, self.__image_size, self.__intrinsics)

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
        return self.__image_renderer(
            camera_w_t_c, chassis_w_t_c, self.__image_size, self.__intrinsics
        ), camera_w_t_c, chassis_w_t_c

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

    def get_state(self) -> Optional[Drone.EState]:
        """
        Try to get the current state of the drone.

        :return:    The current state of the drone, if known, or None otherwise.
        """
        with self.__input_lock:
            return self.__state

    def land(self) -> None:
        """Tell the drone to land."""
        with self.__input_lock:
            if self.__state == Drone.FLYING:
                self.__state = Drone.LANDING

    def move_forward(self, rate: float) -> None:
        """
        Tell the drone to move forward at the specified rate.

        .. note::
            This can also be used to move backwards (by specifying a negative rate).

        :param rate:     The rate at which the drone should move forward (in [-1,1]).
        """
        with self.__input_lock:
            self.__rc_forward = np.clip(rate, -1.0, 1.0)

    def move_right(self, rate: float) -> None:
        """
        Tell the drone to move to the right at the specified rate.

        .. note::
            This can also be used to move to the left (by specifying a negative rate).

        :param rate:    The rate at which the drone should move to the right (in [-1,1]).
        """
        with self.__input_lock:
            self.__rc_right = np.clip(rate, -1.0, 1.0)

    def move_up(self, rate: float) -> None:
        """
        Tell the drone to move up at the specified rate.

        .. note::
            This can also be used to move down (by specifying a negative rate).

        :param rate:    The rate at which the drone should move up (in [-1,1]).
        """
        with self.__input_lock:
            self.__rc_up = np.clip(rate, -1.0, 1.0)

    def set_drone_origin(self, drone_origin: Camera) -> None:
        """
        Set the origin for the drone.

        .. note::
            This can be used to place the drone in a more sensible starting place than the world-space origin.

        :param drone_origin:    The new origin for the drone.
        """
        with self.__input_lock:
            # Update the origin itself.
            self.__drone_origin.set_from(drone_origin)

            # Record that it has changed, so that the drone can be moved to the new origin whenever it is next idle.
            self.__drone_origin_changed.set()

    def set_landing_controller(self, landing_controller: Optional[LandingController]) -> None:
        """
        Set the landing controller to use for the drone.

        :param landing_controller:  The landing controller to use for the drone.
        """
        self.__landing_controller = landing_controller

    def set_takeoff_controller(self, takeoff_controller: Optional[TakeoffController]) -> None:
        """
        Set the takeoff controller to use for the drone.

        :param takeoff_controller:  The takeoff controller to use for the drone.
        """
        self.__takeoff_controller = takeoff_controller

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
            if self.__state == Drone.IDLE:
                self.__state = Drone.TAKING_OFF

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
            self.__rc_yaw = np.clip(rate, -1.0, 1.0)

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
        previous_time: Optional[float] = None

        # Until the simulation should terminate:
        while not self.__should_terminate.is_set():
            # Make a copy of the inputs so that we only need to hold the lock very briefly.
            with self.__input_lock:
                drone_origin: SimpleCamera = CameraUtil.make_default_camera()
                drone_origin.set_from(self.__drone_origin)
                gimbal_pitch: float = self.__gimbal_pitch
                rc_forward: float = self.__rc_forward
                rc_right: float = self.__rc_right
                rc_up: float = self.__rc_up
                rc_yaw: float = self.__rc_yaw
                state: Drone.EState = self.__state

            # TODO: Comment here.
            current_time: float = timer()
            time_offset: Optional[float] = current_time - previous_time if previous_time is not None else None
            previous_time = current_time

            # If the drone's stationary on the ground and its origin has moved:
            if state == Drone.IDLE and self.__drone_origin_changed.is_set():
                # Move the drone to its new origin.
                master_cam.set_from(drone_origin)
                self.__drone_origin_changed.clear()

            # Provided the drone's not stationary on the ground, process any horizontal movements that are requested.
            if state != Drone.IDLE and time_offset is not None:
                master_cam.move_n(time_offset * self.calculate_forward_velocity(rate=rc_forward))
                master_cam.move_u(-time_offset * self.calculate_right_velocity(rate=rc_right))
                master_cam.rotate(master_cam.v(), time_offset * self.calculate_turn_velocity(rate=rc_yaw))

            # Depending on the drone's state:
            if state == Drone.TAKING_OFF:
                # If the drone's taking off, then if a takeoff controller is active, run it; conversely, if no
                # takeoff controller is active, cancel the takeoff.
                if self.__takeoff_controller is not None and time_offset is not None:
                    state = self.__takeoff_controller(master_cam, time_offset)
                else:
                    state = Drone.IDLE
            elif state == Drone.FLYING and time_offset is not None:
                # If the drone's flying, process any vertical movements that are requested.
                master_cam.move_v(time_offset * self.calculate_up_velocity(rate=rc_up))
            elif state == Drone.LANDING:
                # If the drone's landing, then if a landing controller is active, run it; conversely, if no
                # landing controller is active, cancel the landing.
                if self.__landing_controller is not None and time_offset is not None:
                    state = self.__landing_controller(master_cam, time_offset)
                else:
                    state = Drone.FLYING

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
            if state != Drone.IDLE:
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

            # Wait momentarily to avoid a spin loop before processing the next iteration of the simulation.
            time.sleep(0.01)

    # PRIVATE STATIC METHODS

    @staticmethod
    def __calculate_rate(*, units_per_s: float, max_units_per_s: float, allow_clipping: bool) -> Optional[float]:
        """
        TODO

        :param units_per_s:     TODO
        :param max_units_per_s: TODO
        :param allow_clipping:  TODO
        :return:                TODO
        """
        rate: float = units_per_s / max_units_per_s
        if np.fabs(rate) <= 1.0:
            return rate
        elif allow_clipping:
            return float(np.clip(rate, -1.0, 1.0))
        else:
            return None
