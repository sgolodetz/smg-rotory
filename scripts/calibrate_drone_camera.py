import cv2
import glob
import numpy as np

from argparse import ArgumentParser
from typing import Dict

from smg.rotory.drone_factory import DroneFactory


def calibrate_camera() -> None:
    """
    TODO
    """
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    w, h = 9, 6

    per_frame_object_points = np.zeros((h * w, 3), np.float32)
    per_frame_object_points[:, :2] = np.mgrid[0:w, 0:h].T.reshape(-1, 2)

    object_points = []
    image_points = []

    img_filenames = glob.glob("C:/drone_calib/ardrone2_attempt1/*.png")
    img_shape = (0, 0)
    for img_filename in img_filenames:
        print(f"Processing {img_filename}...")

        img = cv2.imread(img_filename)
        grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img_shape = grey.shape[::-1]

        ret, corners = cv2.findChessboardCorners(grey, (w, h), None)

        if ret:
            subpix_corners = cv2.cornerSubPix(grey, corners, (11, 11), (-1, -1), criteria)

            object_points.append(per_frame_object_points)
            image_points.append(subpix_corners)

            corners_img = cv2.drawChessboardCorners(img, (w, h), subpix_corners, ret)
            # cv2.imshow("Corners ({})".format(img_filename), corners_img)
            # cv2.waitKey()
            cv2.imshow("Corners", corners_img)
            cv2.waitKey(1)
            # cv2.destroyAllWindows()
        else:
            print("...could not find chessboard corners")

    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
        object_points, image_points, img_shape, None, None,
        # Tello only
        # flags=cv2.CALIB_ZERO_TANGENT_DIST | cv2.CALIB_FIX_K1 | cv2.CALIB_FIX_K2 | cv2.CALIB_FIX_K3 | cv2.CALIB_FIX_PRINCIPAL_POINT
    )
    print(mtx)
    print(dist)


def save_frames(drone_type: str) -> None:
    """
    TODO

    :param drone_type:  TODO
    """
    kwargs: Dict[str, dict] = {
        "ardrone2": dict(print_commands=False, print_control_messages=False, print_navdata_messages=False),
        "tello": dict(print_commands=False, print_responses=False, print_state_messages=False)
    }

    with DroneFactory.make_drone(drone_type, **kwargs[drone_type]) as drone:
        i: int = 0
        frame_sep: int = 200

        while True:
            image: np.ndarray = drone.get_image()
            cv2.imshow("Image", image)
            if cv2.waitKey(1) == ord('q'):
                break

            if i >= 10 * frame_sep and i % frame_sep == 0:
                cv2.imwrite("C:/drone_calib/{:06d}.png".format(i // frame_sep), image)

            i += 1


def main():
    np.set_printoptions(suppress=True)

    # Parse any command-line arguments.
    parser = ArgumentParser()
    parser.add_argument(
        "--drone_type", "-t", type=str, required=True, choices=("ardrone2", "tello"),
        help="the drone type"
    )
    parser.add_argument(
        "--mode", "-m", type=str, required=True, choices=("calibrate", "save"),
        help="whether to save images prior to calibration, or calibrate the camera now"
    )
    args: dict = vars(parser.parse_args())

    # Either calibrate the camera, or save images prior to calibration, depending on the mode.
    if args["mode"] == "calibrate":
        calibrate_camera()
    else:
        save_frames(args["drone_type"])


if __name__ == "__main__":
    main()
