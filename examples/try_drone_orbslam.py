from contextlib import contextmanager

import cv2
import numpy as np

from argparse import ArgumentParser

from smg.relocalisation.aruco_pnp_relocaliser import ArUcoPnPRelocaliser
from smg.rotory.drone_factory import DroneFactory


# CLASSES

class NullTracker:
    """A null tracker that can be used as a stub for MonocularTracker when it's not available."""

    # PUBLIC METHODS

    # noinspection PyMethodMayBeStatic
    def is_ready(self) -> bool:
        return False

    def terminate(self) -> None:
        pass


# FUNCTIONS

@contextmanager
def make_tracker(drone_type: str):
    """
    Make a monocular ORB-SLAM tracker for a drone, if the package is installed, else make a NullTracker instead.

    .. note::
        This function is intended to be called as the expression of a with statement.
    .. note::
        Since MonocularTracker might or might not be available, I've deliberately avoided specifying the return type.

    :param drone_type:  The type of drone (needed to determine which ORB-SLAM settings to use).
    """
    # Try to create a monocular ORB-SLAM tracker, if the package is installed, else create a NullTracker instead.
    try:
        # noinspection PyUnresolvedReferences
        from smg.pyorbslam2 import MonocularTracker
        tracker = MonocularTracker(
                settings_file=f"settings-{drone_type}.yaml", use_viewer=True,
                voc_file="C:/orbslam2/Vocabulary/ORBvoc.txt", wait_till_ready=False
        )
    except ImportError:
        tracker = NullTracker()

    try:
        # Yield the tracker so that it can be used by the with statement.
        yield tracker
    finally:
        # At the end of the with statement, terminate the tracker.
        tracker.terminate()


# noinspection PyUnresolvedReferences
def main():
    # Parse any command-line arguments.
    parser = ArgumentParser()
    parser.add_argument(
        "--drone_type", "-t", type=str, required=True, choices=("ardrone2", "tello"),
        help="the drone type"
    )
    args: dict = vars(parser.parse_args())

    # Connect to the drone, show the video stream from its camera, and track the pose using ORB-SLAM (if available).
    kwargs: Dict[str, dict] = {
        "ardrone2": dict(print_commands=False, print_control_messages=False, print_navdata_messages=False),
        "tello": dict(print_commands=False, print_responses=False, print_state_messages=False)
    }

    drone_type: str = args.get("drone_type")

    height: float = 1.5
    offset: float = 0.0705
    relocaliser: ArUcoPnPRelocaliser = ArUcoPnPRelocaliser({
        "0_0": np.array([-offset, -(height + offset), 0]),
        "0_1": np.array([offset, -(height + offset), 0]),
        "0_2": np.array([offset, -(height - offset), 0]),
        "0_3": np.array([-offset, -(height - offset), 0])
    })

    with DroneFactory.make_drone(drone_type, **kwargs[drone_type]) as drone:
        with make_tracker(drone_type) as tracker:
            reference_tracker_c_t_w = None
            reference_relocaliser_c_t_w = None

            scale_estimates: List[float] = []

            while True:
                image: np.ndarray = drone.get_image()
                cv2.imshow("Image", image)
                c: int = cv2.waitKey(1)
                if c == ord('q'):
                    break

                if tracker.is_ready():
                    # print("Tracker Pose:")
                    tracker_w_t_c: np.ndarray = tracker.estimate_pose(image)

                    # print("Relocaliser Pose:")
                    relocaliser_c_t_w: np.ndarray = relocaliser.estimate_pose(
                        image, drone.get_intrinsics(), draw_detections=True, print_correspondences=False
                    )

                    if tracker_w_t_c is not None:
                        tracker_c_t_w: np.ndarray = np.linalg.inv(tracker_w_t_c)
                        if relocaliser_c_t_w is not None:
                            # print(tracker_c_t_w[1, 3], relocaliser_c_t_w[1, 3])
                            # print(tracker_c_t_w[0:3, 3], relocaliser_c_t_w[0:3, 3])
                            if reference_relocaliser_c_t_w is not None:
                                tracker_offset = tracker_c_t_w[0:3, 3] - reference_tracker_c_t_w[0:3, 3]
                                relocaliser_offset = relocaliser_c_t_w[0:3, 3] - reference_relocaliser_c_t_w[0:3, 3]
                                min_norm: float = 0.1
                                if np.linalg.norm(relocaliser_offset) >= min_norm:
                                    scale_estimate: float = np.linalg.norm(relocaliser_offset) / np.linalg.norm(tracker_offset)
                                    scale_estimates.append(scale_estimate)
                                    scale: float = np.median(scale_estimates)
                                    print(np.linalg.norm(relocaliser_offset), np.linalg.norm(tracker_offset) * scale, scale_estimate, scale)
                            if c == ord('n'):
                                reference_tracker_c_t_w = tracker_c_t_w
                                reference_relocaliser_c_t_w = relocaliser_c_t_w
                        # else:
                        #     print(tracker_c_t_w[0:3, 3])


if __name__ == "__main__":
    main()
