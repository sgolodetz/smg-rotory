import cv2
import os

os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = "hide"
import pygame

from smg.rotory.drones.tello import Tello
from smg.rotory.joysticks.futaba_t6k import FutabaT6K


def main():
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

    # Use the Futaba T6K to control a DJI Tello.
    with Tello(print_commands=True, print_responses=True, print_state_messages=False) as tello:
        # Stop when both Button 0 and Button 1 on the Futaba T6K are set to their "released" state.
        while joystick.get_button(0) != 0 or joystick.get_button(1) != 0:
            for event in pygame.event.get():
                if event.type == pygame.JOYBUTTONDOWN:
                    # If Button 0 on the Futaba T6K is set to its "pressed" state, take off.
                    if event.button == 0:
                        tello.takeoff()
                elif event.type == pygame.JOYBUTTONUP:
                    # If Button 0 on the Futaba T6K is set to its "released" state, land.
                    if event.button == 0:
                        tello.land()

            # Update the movement of the DJI Tello based on the pitch, roll and yaw values output by the Futaba T6K.
            tello.move_forward(joystick.get_pitch())
            tello.turn(joystick.get_yaw())

            if joystick.get_button(1) == 0:
                tello.move_right(0)
                tello.move_up(joystick.get_roll())
            else:
                tello.move_right(joystick.get_roll())
                tello.move_up(0)

            # Get the most recent image from the DJI Tello and show it.
            cv2.imshow("Tello", tello.get_image())
            cv2.waitKey(1)

    # Shut down pygame cleanly.
    pygame.quit()


if __name__ == "__main__":
    main()
