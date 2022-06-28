import math
import numpy as np
import scipy.optimize
import threading
import time

from collections import defaultdict
from typing import Dict, List, Optional, Set, Tuple

from .beacon import Beacon


class BeaconLocaliser:
    """
    TODO

    See also: https://www.alanzucconi.com/2017/03/13/positioning-and-trilateration/
    """

    # CONSTRUCTOR

    def __init__(self):
        # TODO
        self.__should_terminate: threading.Event = threading.Event()
        self.__test_beacons: Dict[str, Beacon] = {}

        # TODO
        self.__beacon_measurements: Dict[str, List[Tuple[np.ndarray, float]]] = defaultdict(list)
        self.__dirty_beacon_counts: Dict[str, int] = defaultdict(int)
        self.__localised_beacons: Dict[str, Beacon] = {}
        self.__lock: threading.Lock = threading.Lock()

        # TODO
        self.__localisation_thread: threading.Thread = threading.Thread(target=self.__run_localisation)
        self.__localisation_thread.start()

    # PUBLIC STATIC METHODS

    @staticmethod
    def try_localise_beacon(beacon_measurements: List[Tuple[np.ndarray, float]], *, min_needed: int = 3) \
            -> Optional[np.ndarray]:
        # TODO
        if len(beacon_measurements) < min_needed:
            return None

        result: scipy.optimize.optimize.OptimizeResult = scipy.optimize.minimize(
            BeaconLocaliser.__mean_square_error,
            np.zeros(3),
            args=beacon_measurements,
            method="L-BFGS-B",
            options={
                "ftol": 1e-5,
                "maxiter": 1e+7
            }
        )

        return result.x

    # PUBLIC METHODS

    def add_beacon_measurements(self, receiver_pos: np.ndarray, beacon_ranges: Dict[str, float]) -> None:
        # TODO
        with self.__lock:
            for beacon_name, beacon_range in beacon_ranges.items():
                self.__beacon_measurements[beacon_name].append((receiver_pos, beacon_range))
                self.__beacon_measurements[beacon_name] = self.__beacon_measurements[beacon_name][-50:]
                self.__dirty_beacon_counts[beacon_name] += 1

        # print(self.__beacon_measurements)
        # if "Foo" in self.__beacon_measurements:
        #     print(len(self.__beacon_measurements["Foo"]))

    def get_beacons(self) -> Dict[str, Beacon]:
        # TODO
        with self.__lock:
            return {**self.__localised_beacons, **self.get_test_beacons()}

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
        with self.__lock:
            if beacon_name in self.__beacon_measurements:
                del self.__beacon_measurements[beacon_name]

    def terminate(self) -> None:
        # TODO
        if not self.__should_terminate.is_set():
            self.__should_terminate.set()

            # Wait for the localisation thread to terminate.
            self.__localisation_thread.join()

    # PRIVATE METHODS

    def __run_localisation(self) -> None:
        # TODO
        while not self.__should_terminate.is_set():
            # Make a copy of the shared variables so that we only need to hold the lock very briefly.
            with self.__lock:
                beacon_measurements: Dict[str, List[Tuple[np.ndarray, float]]] = self.__beacon_measurements.copy()
                dirty_beacon_counts: Dict[str, int] = self.__dirty_beacon_counts.copy()

            for beacon_name in dirty_beacon_counts:
                measurements_for_beacon: Optional[List[Tuple[np.ndarray, float]]] = beacon_measurements.get(beacon_name)
                if measurements_for_beacon is None:
                    continue

                beacon_pos: Optional[np.ndarray] = BeaconLocaliser.try_localise_beacon(measurements_for_beacon)

                if beacon_pos is not None:
                    with self.__lock:
                        self.__localised_beacons[f"L_{beacon_name}"] = Beacon(beacon_pos, 1.0)
                        self.__dirty_beacon_counts[beacon_name] -= 1
                        if self.__dirty_beacon_counts[beacon_name] == 0:
                            del self.__dirty_beacon_counts[beacon_name]

                    # print(f"Beacon {beacon_name} localised at {beacon_pos}!", self.__test_beacons["Foo"].position)

            # Wait momentarily to avoid a spin loop.
            time.sleep(0.01)

    # PRIVATE STATIC METHODS

    @staticmethod
    def __mean_square_error(beacon_pos: np.ndarray, beacon_measurements: List[Tuple[np.ndarray, float]]) -> float:
        # TODO
        mse: float = 0.0
        for receiver_pos, measured_distance in beacon_measurements:
            calculated_distance: float = np.linalg.norm(receiver_pos - beacon_pos)
            mse += math.pow(calculated_distance - measured_distance, 2.0)

        return mse / len(beacon_measurements)
