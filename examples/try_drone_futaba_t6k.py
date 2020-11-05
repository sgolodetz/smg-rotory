import cv2
import os

os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = "hide"
import pygame

from argparse import ArgumentParser

from smg.rotory.drone_factory import DroneFactory
from smg.rotory.joysticks.futaba_t6k import FutabaT6K


def main():
    # Parse any command-line arguments.
    parser = ArgumentParser()
    parser.add_argument(
        "--drone_type", "-t", type=str, required=True, choices=("ardrone2", "tello"),
        help="the drone type"
    )
    args: dict = vars(parser.parse_args())

    # Initialise pygame and its joystick module.
    pygame.init()
    pygame.joystick.init()

    # Try to determine the joystick index of the Futaba T6K. If no joystick is plugged in, early out.
    joystick_count = pygame.joystick.get_count()
    joystick_idx = 0
    if joystick_count == 0:
        exit(0)
    elif joystick_count != 1:
        # TODO: Prompt the user for the joystick to use.
        pass

    # Construct and calibrate the Futaba T6K.
    joystick = FutabaT6K(joystick_idx)
    joystick.calibrate()

    # Use the Futaba T6K to control a drone.
    with DroneFactory.make_drone(args.get("drone_type")) as drone:
        # Stop when both Button 0 and Button 1 on the Futaba T6K are set to their "released" state.
        while joystick.get_button(0) != 0 or joystick.get_button(1) != 0:
            for event in pygame.event.get():
                if event.type == pygame.JOYBUTTONDOWN:
                    # If Button 0 on the Futaba T6K is set to its "pressed" state, take off.
                    if event.button == 0:
                        drone.takeoff()
                elif event.type == pygame.JOYBUTTONUP:
                    # If Button 0 on the Futaba T6K is set to its "released" state, land.
                    if event.button == 0:
                        drone.land()

            # Update the movement of the drone based on the pitch, roll and yaw values output by the Futaba T6K.
            drone.move_forward(joystick.get_pitch())
            drone.turn(joystick.get_yaw())

            if joystick.get_button(1) == 0:
                drone.move_right(0)
                drone.move_up(joystick.get_roll())
            else:
                drone.move_right(joystick.get_roll())
                drone.move_up(0)

            # Get the most recent image from the drone and show it.
            cv2.imshow("Image", drone.get_image())
            cv2.waitKey(1)

    # Shut down pygame cleanly.
    pygame.quit()


if __name__ == "__main__":
    main()
