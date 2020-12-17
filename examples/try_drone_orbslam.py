from contextlib import contextmanager

import cv2
import numpy as np

from argparse import ArgumentParser
from typing import Dict

from smg.rotory import DroneFactory


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

    with DroneFactory.make_drone(drone_type, **kwargs[drone_type]) as drone:
        with make_tracker(drone_type) as tracker:
            while True:
                image: np.ndarray = drone.get_image()
                cv2.imshow("Image", image)
                if cv2.waitKey(1) == ord('q'):
                    break

                if tracker.is_ready():
                    print(tracker.estimate_pose(image))


if __name__ == "__main__":
    main()
