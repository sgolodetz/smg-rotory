from smg.rotory.drones.ardrone2 import ARDrone2


def main():
    # Connect to the drone, and then repeatedly print out some telemetry.
    with ARDrone2(print_commands=False, print_control_messages=False, print_navdata_messages=False) as drone:
        while True:
            print(f"Time: {drone.get_time()}; Orientation: {drone.get_orientation()}")
            print(drone.get_navdata_option_fields("altitude"))
            print(drone.get_navdata_option_fields("euler_angles"))
            print(drone.get_navdata_option_fields("gyros_offsets"))
            print(drone.get_navdata_option_fields("phys_measures"))
            print(drone.get_navdata_option_fields("raw_measures"))
            print(drone.get_navdata_option_fields("trims"))
            print(drone.get_navdata_option_fields("watchdog"))


if __name__ == "__main__":
    main()
