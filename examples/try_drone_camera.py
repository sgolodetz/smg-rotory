import cv2
import numpy as np

from argparse import ArgumentParser
from typing import Dict

from smg.rotory.drone_factory import DroneFactory


def main():
    # Parse any command-line arguments.
    parser = ArgumentParser()
    parser.add_argument(
        "--drone_type", "-t", type=str, required=True, choices=("ardrone2", "tello"),
        help="the drone type"
    )
    args: dict = vars(parser.parse_args())

    # Connect to the drone, and then show the video stream from its camera.
    kwargs: Dict[str, dict] = {
        "ardrone2": dict(
            camera_matrices=[
                np.array([
                    [545.0907676, 0., 320.83246651],
                    [0., 540.66721899, 162.46881425],
                    [0., 0., 1.]
                ]),
                None
            ],
            dist_coeffs=[
                np.array([[-0.51839052, 0.58636131, 0.0037668, -0.00869583, -0.65549135]]),
                None
            ],
            print_commands=True,
            print_control_messages=True,
            print_navdata_messages=False
        ),
        "tello": dict(
            print_commands=True,
            print_responses=True,
            print_state_messages=False
        )
    }

    drone_type: str = args.get("drone_type")

    with DroneFactory.make_drone(drone_type, **kwargs[drone_type]) as drone:
        delay_time: int = 0
        while True:
            image: np.ndarray = drone.get_image()
            cv2.imshow("Image", image)

            c: int = cv2.waitKey(1)
            if c == ord('q'):
                break
            elif c == ord('s') and delay_time == 0:
                drone.switch_camera()
                delay_time = 100

            if delay_time > 0:
                delay_time -= 1


if __name__ == "__main__":
    main()
