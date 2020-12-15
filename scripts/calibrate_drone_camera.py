import cv2
import glob
import numpy as np
import os

from argparse import ArgumentParser
from typing import Dict, Optional, Tuple

from smg.rotory import DroneFactory


def calibrate_camera(drone_type: str, image_dir: str) -> None:
    """
    Calibrate the camera of a drone of the specified type, using calibration images saved in the specified directory.

    .. note::
        See opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_calib3d/py_calibration/py_calibration.html.

    :param drone_type:  The type of drone.
    :param image_dir:   The directory containing the calibration images.
    """
    # Specify the size of the calibration chessboard.
    w, h = 9, 6

    # Setup the 3D object points that will be used for each image.
    per_image_object_points = np.zeros((h * w, 3), np.float32)
    per_image_object_points[:, :2] = np.mgrid[0:w, 0:h].T.reshape(-1, 2)

    # Setup lists in which to store the 3D object points and 2D image points for all images.
    object_points = []
    image_points = []

    # For each image in the image directory:
    image_filenames = glob.glob(f"{image_dir}/*.png")
    image_shape: Tuple[int, int] = (0, 0)

    for image_filename in image_filenames:
        print(f"Processing {image_filename}...")

        # Load in the image, and record its shape (this only needs doing once, since all images have the same shape).
        image: np.ndarray = cv2.imread(image_filename)
        image_shape = image.shape[:2][::-1]

        # Try to find the calibration chessboard corners in the image.
        result = find_corners(image, w, h)

        if result:
            # If successful, update the points lists, and visualise the chessboard corners.
            corners, subpix_corners = result

            object_points.append(per_image_object_points)
            image_points.append(subpix_corners)

            corners_image: np.ndarray = cv2.drawChessboardCorners(image, (w, h), subpix_corners, True)
            cv2.imshow("Corners", corners_image)
            cv2.waitKey(1)
        else:
            print("...could not find chessboard corners")

    # Calibrate the camera. If we're using a Tello drone, we fix some of the camera parameters, on the basis that
    # images from the Tello camera don't seem to suffer from any distortion.
    flags: int = 0
    # if drone_type == "tello":
    #     flags = cv2.CALIB_ZERO_TANGENT_DIST | cv2.CALIB_FIX_K1 | cv2.CALIB_FIX_K2 | cv2.CALIB_FIX_K3 | \
    #             cv2.CALIB_FIX_PRINCIPAL_POINT

    ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
        object_points, image_points, image_shape, None, None, flags=flags
    )

    # Print out the results of the calibration process.
    print(camera_matrix)
    print(dist_coeffs)

    # Destroy all of the OpenCV windows we were using during the calibration.
    cv2.destroyAllWindows()

    # For each image in the image directory:
    for image_filename in image_filenames:
        print(f"Processing {image_filename}...")

        # Load in the image and undistort it.
        distorted_image: np.ndarray = cv2.imread(image_filename)
        undistorted_image: np.ndarray = cv2.undistort(distorted_image, camera_matrix, dist_coeffs)

        # Try to visualise the chessboard corners on the distorted and undistorted images.
        distorted_corners: Optional[np.ndarray] = make_corners_image(distorted_image, w, h)
        undistorted_corners: Optional[np.ndarray] = make_corners_image(undistorted_image, w, h)

        cv2.imshow("Distorted Image", distorted_corners if distorted_corners is not None else distorted_image)
        cv2.imshow("Undistorted Image", undistorted_corners if undistorted_corners is not None else undistorted_image)

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
    # Make sure the image directory exists.
    os.makedirs(image_dir, exist_ok=True)

    # If it already existed and was non-empty, early out so as to avoid overwriting existing files.
    if len(os.listdir(image_dir)) != 0:
        raise ValueError(f"Won't save calibration images into non-empty directory '{image_dir}' for safety reasons")

    # Connect to the drone, and save images to the image directory at regular intervals.
    kwargs: Dict[str, dict] = {
        "ardrone2": dict(print_commands=False, print_control_messages=False, print_navdata_messages=False),
        "tello": dict(print_commands=False, print_responses=False, print_state_messages=False)
    }

    with DroneFactory.make_drone(drone_type, **kwargs[drone_type]) as drone:
        i: int = 0
        image_sep: int = 200

        while True:
            image: np.ndarray = drone.get_image()
            cv2.imshow("Image", image)

            if cv2.waitKey(1) == ord('q'):
                break

            if i >= 10 * image_sep and i % image_sep == 0:
                cv2.imwrite(f"{image_dir}/{i // image_sep:06d}.png", image)

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
