import cv2
import numpy as np

from argparse import ArgumentParser

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
    with DroneFactory.make_drone(args.get("drone_type")) as drone:
        while True:
            image: np.ndarray = drone.get_image()
            cv2.imshow("Image", image)
            if cv2.waitKey(1) == ord('q'):
                break


if __name__ == "__main__":
    main()
