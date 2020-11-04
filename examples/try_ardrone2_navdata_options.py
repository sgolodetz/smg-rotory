import cv2
import numpy as np
import struct

from typing import Optional

from smg.rotory.drones.ardrone2 import ARDrone2
from smg.rotory.util.bits_util import BitsUtil


def main():
    # Connect to the drone, and then show the video stream from its camera.
    with ARDrone2(print_commands=False, print_control_messages=False, print_navdata_messages=False) as drone:
        while True:
            image: np.ndarray = drone.get_image()
            cv2.imshow("Image", image)
            if cv2.waitKey(1) == ord('q'):
                break

            time: Optional[bytes] = drone.get_navdata_option("time")
            if time is not None and len(time) == 4:
                time_bits: str = f"{struct.unpack('i', time)[0]:032b}"
                secs: int = BitsUtil.convert_hilo_bit_string_to_int32(time_bits[:11])
                microsecs: int = BitsUtil.convert_hilo_bit_string_to_int32(time_bits[11:])
                print(f"Time: {(secs, microsecs)}")


if __name__ == "__main__":
    main()
