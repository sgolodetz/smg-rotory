from typing import Dict, Optional

from .beacon import Beacon


class BeaconLocaliser:
    """TODO"""

    # CONSTRUCTOR

    def __init__(self):
        # TODO
        self.__test_beacons: Dict[str, Beacon] = {}

    # PUBLIC METHODS

    def add_beacon_ranges(self, beacon_ranges: Dict[str, float]) -> None:
        # TODO
        pass

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
