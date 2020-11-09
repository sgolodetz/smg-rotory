import cv2
import glob
import numpy as np
import os

from argparse import ArgumentParser
from typing import Dict, Optional, Tuple

from smg.rotory.drone_factory import DroneFactory


def calibrate_camera(drone_type: str, image_dir: str) -> None:
    """
    Calibrate the camera of a drone of the specified type, using calibration images saved in the specified directory.

    :param drone_type:  The type of drone.
    :param image_dir:   The directory containing the calibration images.
    """
    # Specify the size of the calibration chessboard.
    w, h = 9, 6

    per_image_object_points = np.zeros((h * w, 3), np.float32)
    per_image_object_points[:, :2] = np.mgrid[0:w, 0:h].T.reshape(-1, 2)

    object_points = []
    image_points = []

    image_filenames = glob.glob(f"{image_dir}/*.png")
    image_shape = (0, 0)
    for image_filename in image_filenames:
        print(f"Processing {image_filename}...")

        image = cv2.imread(image_filename)
        image_shape = image.shape[:2][::-1]

        result = find_corners(image, w, h)
        if result:
            corners, subpix_corners = result

            object_points.append(per_image_object_points)
            image_points.append(subpix_corners)

            corners_img = cv2.drawChessboardCorners(image, (w, h), subpix_corners, True)
            cv2.imshow("Corners", corners_img)
            cv2.waitKey(1)
        else:
            print("...could not find chessboard corners")

    flags: int = 0
    if drone_type == "tello":
        flags = cv2.CALIB_ZERO_TANGENT_DIST | cv2.CALIB_FIX_K1 | cv2.CALIB_FIX_K2 | cv2.CALIB_FIX_K3 | \
                cv2.CALIB_FIX_PRINCIPAL_POINT

    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(object_points, image_points, image_shape, None, None, flags=flags)
    print(mtx)
    print(dist)

    cv2.destroyAllWindows()

    for image_filename in image_filenames:
        print(f"Processing {image_filename}...")

        image = cv2.imread(image_filename)
        undistorted_img = cv2.undistort(image, mtx, dist)

        corners_img = make_corners_image(image, w, h)
        undistorted_corners_img = make_corners_image(undistorted_img, w, h)

        cv2.imshow("Image", corners_img if corners_img is not None else image)
        cv2.imshow("Undistorted Image", undistorted_corners_img if undistorted_corners_img is not None else undistorted_img)
        if cv2.waitKey() == ord('q'):
            break


def find_corners(image: np.ndarray, w: int, h: int) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """
    Try to find the calibration chessboard corners in the input image.

    :param image:   The input image.
    :param w:       The width of the chessboard pattern.
    :param h:       The height of the chessboard pattern.
    :return:        A tuple consisting of the corners and the refined corners, if found, or None otherwise.
    """
    grey = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    ret, corners = cv2.findChessboardCorners(grey, (w, h), None)
    if ret:
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        subpix_corners = cv2.cornerSubPix(grey, corners, (11, 11), (-1, -1), criteria)
        return corners, subpix_corners
    else:
        return None


def make_corners_image(image: np.ndarray, w: int, h: int) -> Optional[np.ndarray]:
    """
    Try to make a version of the input image onto which the calibration chessboard corners have been superimposed.

    :param image:   The input image.
    :param w:       The width of the chessboard pattern.
    :param h:       The height of the chessboard pattern.
    :return:        A version of the input image onto which the chessboard corners have been superimposed,
                    if they were successfully found, or None otherwise.
    """
    result = find_corners(image, w, h)
    if result:
        corners, subpix_corners = result
        return cv2.drawChessboardCorners(image, (w, h), subpix_corners, True)
    else:
        return None


def save_images(drone_type: str, image_dir: str) -> None:
    """
    Save images from a drone of the specified type for later calibration.

    :param drone_type:  The type of drone.
    :param image_dir:   The directory to which to save the calibration images.
    """
    os.makedirs(image_dir, exist_ok=True)
    if len(os.listdir(image_dir)) != 0:
        raise ValueError(f"Cannot save calibration images into non-empty directory '{image_dir}'")

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
                cv2.imwrite(f"{image_dir}/{i // frame_sep:06d}.png", image)

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
        "--image_dir", "-d", type=str, required=True,
        help="the directory {to/from} which to {save/load} the calibration images"
    )
    parser.add_argument(
        "--mode", "-m", type=str, required=True, choices=("calibrate", "save"),
        help="whether to save images for later calibration, or calibrate the camera now"
    )
    args: dict = vars(parser.parse_args())

    # Either calibrate the camera, or save images for later calibration, depending on the mode.
    drone_type: str = args["drone_type"]
    image_dir: str = args["image_dir"]

    if args["mode"] == "calibrate":
        calibrate_camera(drone_type, image_dir)
    else:
        save_images(drone_type, image_dir)


if __name__ == "__main__":
    main()
