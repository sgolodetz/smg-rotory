import time

from argparse import ArgumentParser

from smg.rotory import DroneFactory


def main():
    # Parse any command-line arguments.
    parser = ArgumentParser()
    parser.add_argument(
        "--drone_type", "-t", type=str, required=True, choices=("ardrone2", "tello"),
        help="the drone type"
    )
    args: dict = vars(parser.parse_args())

    # Connect to the drone, then tell it to take off, wait 5 seconds and land.
    with DroneFactory.make_drone(args.get("drone_type")) as drone:
        drone.takeoff()
        time.sleep(5)
        drone.land()


if __name__ == "__main__":
    main()
