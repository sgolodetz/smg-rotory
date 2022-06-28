import numpy as np

from collections import defaultdict
from typing import Dict, List, Optional, Tuple

from .beacon import Beacon


class BeaconLocaliser:
    """TODO"""

    # CONSTRUCTOR

    def __init__(self):
        # TODO
        self.__beacon_measurements: Dict[str, List[Tuple[np.ndarray, float]]] = defaultdict(list)
        self.__test_beacons: Dict[str, Beacon] = {}

    # PUBLIC METHODS

    def add_beacon_measurements(self, receiver_pos: np.ndarray, beacon_ranges: Dict[str, float]) -> None:
        # TODO
        for beacon_name, beacon_range in beacon_ranges.items():
            self.__beacon_measurements[beacon_name].append((receiver_pos, beacon_range))

        print(self.__beacon_measurements)
        if "Foo" in self.__beacon_measurements:
            print(len(self.__beacon_measurements["Foo"]))

    def get_beacons(self) -> Dict[str, Beacon]:
        # TODO
        # FIXME: Temporary.
        return self.get_test_beacons()

    def get_test_beacons(self) -> Dict[str, Beacon]:
        # TODO
        return self.__test_beacons

    def set_test_beacon(self, beacon_name: str, beacon: Optional[Beacon]) -> None:
        # TODO
        if beacon is not None:
            self.__test_beacons[beacon_name] = beacon
        elif beacon_name in self.__test_beacons:
            del self.__test_beacons[beacon_name]

        # TODO
        if beacon_name in self.__beacon_measurements:
            del self.__beacon_measurements[beacon_name]
