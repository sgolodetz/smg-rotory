import cv2
import numpy as np

from argparse import ArgumentParser
from typing import Dict, List

from smg.pyorbslam2 import MonocularTracker
from smg.relocalisation import ArUcoPnPRelocaliser
from smg.rotory import DroneFactory


def main():
    np.set_printoptions(suppress=True)

    # Parse any command-line arguments.
    parser = ArgumentParser()
    parser.add_argument(
        "--drone_type", "-t", type=str, required=True, choices=("ardrone2", "tello"),
        help="the drone type"
    )
    args: dict = vars(parser.parse_args())

    # Set up a relocaliser that uses an ArUco marker of a known size and at a known height to relocalise.
    height: float = 1.285  # 1.285m (the height of the centre of the printed marker)
    offset: float = 0.0705  # 7.05cm (half the width of the printed marker)
    relocaliser: ArUcoPnPRelocaliser = ArUcoPnPRelocaliser({
        "0_0": np.array([-offset, -(height + offset), 0]),
        "0_1": np.array([offset, -(height + offset), 0]),
        "0_2": np.array([offset, -(height - offset), 0]),
        "0_3": np.array([-offset, -(height - offset), 0])
    })

    # Connect to the drone.
    kwargs: Dict[str, dict] = {
        "ardrone2": dict(print_commands=False, print_control_messages=False, print_navdata_messages=False),
        "tello": dict(print_commands=False, print_responses=False, print_state_messages=False)
    }

    drone_type: str = args.get("drone_type")

    with DroneFactory.make_drone(drone_type, **kwargs[drone_type]) as drone:
        # Track the pose of the drone using ORB-SLAM.
        with MonocularTracker(
            settings_file=f"settings-{drone_type}.yaml", use_viewer=True,
            voc_file="C:/orbslam2/Vocabulary/ORBvoc.txt", wait_till_ready=False
        ) as tracker:
            reference_tracker_w_t_c = None
            reference_relocaliser_w_t_c = None

            scale: float = 1.0
            scale_estimates: List[float] = []
            showing_poses = False

            while True:
                image: np.ndarray = drone.get_image()
                cv2.imshow("Image", image)
                c: int = cv2.waitKey(1)
                if c == ord('q'):
                    break

                if tracker.is_ready():
                    tracker_c_t_w: np.ndarray = tracker.estimate_pose(image)

                    relocaliser_w_t_c: np.ndarray = relocaliser.estimate_pose(
                        image, drone.get_intrinsics(), draw_detections=True, print_correspondences=False
                    )

                    if tracker_c_t_w is not None:
                        tracker_w_t_c: np.ndarray = np.linalg.inv(tracker_c_t_w)

                        if c == ord('m') or showing_poses:
                            showing_poses = True
                            print("===BEGIN===")
                            if relocaliser_w_t_c is not None:
                                print("Relocaliser Pose:")
                                print(relocaliser_w_t_c)
                            if reference_tracker_w_t_c is not None:
                                print("Tracker Pose:")
                                scaled_reference_tracker_w_t_c: np.ndarray = reference_tracker_w_t_c.copy()
                                scaled_reference_tracker_w_t_c[0:3, :] *= scale
                                scaled_tracker_w_t_c: np.ndarray = tracker_w_t_c.copy()
                                scaled_tracker_w_t_c[0:3, :] *= scale
                                print(scaled_tracker_w_t_c @ np.linalg.inv(scaled_reference_tracker_w_t_c) @ reference_relocaliser_w_t_c)
                            print("===END===")
                        elif relocaliser_w_t_c is not None:
                            if c == ord('n'):
                                reference_tracker_w_t_c = tracker_w_t_c
                                reference_relocaliser_w_t_c = relocaliser_w_t_c
                                scale_estimates.clear()
                                scale = 1.0

                            if reference_relocaliser_w_t_c is not None:
                                tracker_offset = tracker_w_t_c[0:3, 3] - reference_tracker_w_t_c[0:3, 3]
                                relocaliser_offset = relocaliser_w_t_c[0:3, 3] - reference_relocaliser_w_t_c[0:3, 3]
                                min_norm: float = 0.1
                                if np.linalg.norm(relocaliser_offset) >= min_norm:
                                    scale_estimate: float = np.linalg.norm(relocaliser_offset) / np.linalg.norm(tracker_offset)
                                    scale_estimates.append(scale_estimate)
                                    scale = np.median(scale_estimates)
                                    print(np.linalg.norm(relocaliser_offset), np.linalg.norm(tracker_offset) * scale, scale_estimate, scale)


if __name__ == "__main__":
    main()
