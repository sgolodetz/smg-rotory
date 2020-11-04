from smg.rotory.drones.ardrone2 import ARDrone2


def main():
    # Connect to the drone, and then repeatedly print out the time reported by the navdata.
    with ARDrone2(print_commands=False, print_control_messages=False, print_navdata_messages=False) as drone:
        while True:
            print(f"Time: {drone.get_time()}")


if __name__ == "__main__":
    main()
