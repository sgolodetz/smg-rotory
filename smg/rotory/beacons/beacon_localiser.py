import copy
import math
import numpy as np
import operator
import random
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
        self.__fake_beacons: Dict[str, Beacon] = {}
        self.__rng: random.Random = random.Random(12345)
        self.__should_terminate: threading.Event = threading.Event()

        # TODO
        self.__beacon_measurements: Dict[str, List[Tuple[np.ndarray, float]]] = defaultdict(list)
        self.__dirty_beacons: Set[str] = set()
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
            # beacon_measurements[0][0],
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
                if len(self.__beacon_measurements[beacon_name]) == 0 \
                        or np.linalg.norm(receiver_pos - self.__beacon_measurements[beacon_name][-1][0]) > 0.01:
                    beacon_measurement: Tuple[np.ndarray, float] = (receiver_pos, beacon_range)
                    if len(self.__beacon_measurements[beacon_name]) == 50:
                        idx: int = self.__rng.randrange(0, len(self.__beacon_measurements[beacon_name]) - 1)
                        self.__beacon_measurements[beacon_name][idx] = self.__beacon_measurements[beacon_name][-1]
                        self.__beacon_measurements[beacon_name][-1] = beacon_measurement
                    else:
                        self.__beacon_measurements[beacon_name].append(beacon_measurement)
                    self.__dirty_beacons.add(beacon_name)

        # print(self.__beacon_measurements)
        # if "Foo" in self.__beacon_measurements:
        #     print(len(self.__beacon_measurements["Foo"]))

    def get_beacon_measurements(self) -> Dict[str, List[Tuple[np.ndarray, float]]]:
        with self.__lock:
            return copy.deepcopy(self.__beacon_measurements)

    def get_beacons(self) -> Dict[str, Beacon]:
        # TODO
        with self.__lock:
            return {**copy.deepcopy(self.__localised_beacons), **self.get_fake_beacons()}

    def get_fake_beacons(self) -> Dict[str, Beacon]:
        # TODO
        return self.__fake_beacons

    def set_fake_beacon(self, beacon_name: str, beacon: Optional[Beacon]) -> None:
        # TODO
        if beacon is not None:
            self.__fake_beacons[beacon_name] = beacon
        elif beacon_name in self.__fake_beacons:
            del self.__fake_beacons[beacon_name]

        # TODO
        with self.__lock:
            if beacon_name in self.__beacon_measurements:
                del self.__beacon_measurements[beacon_name]

            if f"L_{beacon_name}" in self.__localised_beacons:
                del self.__localised_beacons[f"L_{beacon_name}"]

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
                beacon_measurements: Dict[str, List[Tuple[np.ndarray, float]]] = copy.deepcopy(
                    self.__beacon_measurements
                )
                dirty_beacons: Set[str] = self.__dirty_beacons.copy()
                self.__dirty_beacons.clear()

            for beacon_name in dirty_beacons:
                measurements_for_beacon: Optional[List[Tuple[np.ndarray, float]]] = beacon_measurements.get(beacon_name)
                if measurements_for_beacon is None:
                    continue

                beacon_pos: Optional[np.ndarray] = BeaconLocaliser.try_localise_beacon(measurements_for_beacon)

                if beacon_pos is not None:
                    with self.__lock:
                        max_range: float = max(measurements_for_beacon, key=operator.itemgetter(1))[1]
                        self.__localised_beacons[f"L_{beacon_name}"] = Beacon(
                            beacon_pos, max_range, Beacon.BT_LOCALISED
                        )

                    print(f"Beacon {beacon_name} localised at {beacon_pos}!", self.__fake_beacons["Foo"].position)

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
