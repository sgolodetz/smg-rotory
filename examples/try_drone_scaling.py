import cv2
import numpy as np

from argparse import ArgumentParser
from typing import Dict, Optional

from smg.pyorbslam2 import MonocularTracker
from smg.relocalisation import ArUcoPnPRelocaliser
from smg.relocalisation.poseglobalisers import MonocularPoseGlobaliser
from smg.rotory import DroneFactory


def print_frame_poses(tracker_w_t_c: np.ndarray, relocaliser_w_t_c: Optional[np.ndarray]):
    print("===BEGIN FRAME===")
    if relocaliser_w_t_c is not None:
        print("Relocaliser Pose:")
        print(relocaliser_w_t_c)
    print("Tracker Pose:")
    print(tracker_w_t_c)
    print("===END FRAME===")


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
    height: float = 1.5  # 1.5m (the height of the centre of the printed marker)
    offset: float = 0.0705  # 7.05cm (half the width of the printed marker)
    relocaliser: ArUcoPnPRelocaliser = ArUcoPnPRelocaliser({
        "0_0": np.array([-offset, -(height + offset), 0]),
        "0_1": np.array([offset, -(height + offset), 0]),
        "0_2": np.array([offset, -(height - offset), 0]),
        "0_3": np.array([-offset, -(height - offset), 0])
    })

    # Construct a monocular pose globaliser.
    pose_globaliser: MonocularPoseGlobaliser = MonocularPoseGlobaliser(debug=True)

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
            while True:
                image: np.ndarray = drone.get_image()
                cv2.imshow("Image", image)
                c: int = cv2.waitKey(1)
                if c == ord('q'):
                    break

                if tracker.is_ready():
                    tracker_c_t_i: np.ndarray = tracker.estimate_pose(image)
                    if tracker_c_t_i is None:
                        continue

                    tracker_i_t_c: np.ndarray = np.linalg.inv(tracker_c_t_i)

                    relocaliser_w_t_c: Optional[np.ndarray] = relocaliser.estimate_pose(
                        image, drone.get_intrinsics(), draw_detections=True, print_correspondences=False
                    )

                    state: MonocularPoseGlobaliser.EState = pose_globaliser.get_state()

                    if state == MonocularPoseGlobaliser.ACTIVE:
                        tracker_w_t_c: np.ndarray = pose_globaliser.apply(tracker_i_t_c)
                        print_frame_poses(tracker_w_t_c, relocaliser_w_t_c)
                        if c == ord('f'):
                            pose_globaliser.set_fixed_height(tracker_w_t_c)
                    elif state == MonocularPoseGlobaliser.TRAINING:
                        if relocaliser_w_t_c is not None:
                            pose_globaliser.train(tracker_i_t_c, relocaliser_w_t_c)
                        if c == ord('m'):
                            pose_globaliser.finish_training()
                    elif state == MonocularPoseGlobaliser.UNTRAINED:
                        if c == ord('n') and relocaliser_w_t_c is not None:
                            pose_globaliser.start_training(tracker_i_t_c, relocaliser_w_t_c)


if __name__ == "__main__":
    main()
