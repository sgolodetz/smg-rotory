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
    Used to localise beacons based on measurements of the ranges to them from different positions.

    See also: https://www.alanzucconi.com/2017/03/13/positioning-and-trilateration/
    """

    # CONSTRUCTOR

    def __init__(self, *, debug: bool = True, max_measurements_per_beacon: int = 50,
                 min_receiver_movement: float = 0.01):
        """
        Construct a beacon localiser.

        :param debug:                       Whether or not to enable debugging.
        :param max_measurements_per_beacon: The maximum number of measurements we want to retain for each beacon.
        :param min_receiver_movement:       The minimum distance (in m) that the receiver must have moved since the
                                            last measurement for a beacon before a new measurement can be accepted.
        """
        self.__debug: bool = debug
        self.__fake_beacons: Dict[str, Beacon] = {}
        self.__max_measurements_per_beacon: int = max_measurements_per_beacon
        self.__min_receiver_movement: float = min_receiver_movement
        self.__rng: random.Random = random.Random(12345)
        self.__should_terminate: threading.Event = threading.Event()

        # The shared variables, together with their lock.
        self.__beacon_measurements: Dict[str, List[Tuple[np.ndarray, float]]] = defaultdict(list)
        self.__dirty_beacons: Set[str] = set()
        self.__localised_beacons: Dict[str, Beacon] = {}
        self.__lock: threading.Lock = threading.Lock()

        # Set up and start the beacon localisation thread.
        self.__localisation_thread: threading.Thread = threading.Thread(target=self.__run_localisation)
        self.__localisation_thread.start()

    # PUBLIC STATIC METHODS

    @staticmethod
    def try_localise_beacon(beacon_measurements: List[Tuple[np.ndarray, float]], *,
                            min_needed: int = 3) -> Optional[np.ndarray]:
        """
        Try to localise a beacon based on measurements of the range to it from a number of different positions.

        :param beacon_measurements: The range measurements, as a list of (receiver position, beacon range) tuples.
        :param min_needed:          The minimum number of measurements needed for localisation to be attempted.
        :return:                    The estimated location of the beacon, if successful, or None otherwise.
        """
        # If there are too few measurements for localisation to be attempted, early out.
        if len(beacon_measurements) < min_needed:
            return None

        # Try to localise the beacon.
        result: scipy.optimize.optimize.OptimizeResult = scipy.optimize.minimize(
            BeaconLocaliser.__mean_square_error,
            np.zeros(3),  # beacon_measurements[0][0],
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
        """
        Add to the localiser some measurements of the ranges to different beacons from the specified receiver position.

        .. note::
            The idea is that the beacons are transmitters, and that at each time step, the receiver picks up a signal
            from each beacon that can be used to estimate its range. We can thus accumulate multiple measurements for
            each beacon over time, from different receiver positions. In practice, we limit the number of measurements
            that we retain for each beacon to keep the localisation problem we plan to solve tractable (and to avoid
            unbounded memory usage).

        :param receiver_pos:    The position of the receiver.
        :param beacon_ranges:   A dictionary that maps the names of the beacons to their measured ranges (in m).
        """
        with self.__lock:
            # For each beacon for which a measured range is available:
            for beacon_name, beacon_range in beacon_ranges.items():
                # If either (i) there are no existing measurements for the beacon, or (ii) the receiver has moved
                # at least a short distance since the most recent measurement for the beacon:
                if len(self.__beacon_measurements[beacon_name]) == 0 or np.linalg.norm(
                            receiver_pos - self.__beacon_measurements[beacon_name][-1][0]
                        ) > self.__min_receiver_movement:
                    # Construct the new measurement.
                    beacon_measurement: Tuple[np.ndarray, float] = (receiver_pos, beacon_range)

                    # If we already have the maximum number of measurements we want to retain for this beacon:
                    if len(self.__beacon_measurements[beacon_name]) == self.__max_measurements_per_beacon:
                        # Randomly pick an existing measurement to discard.
                        idx: int = self.__rng.randrange(0, len(self.__beacon_measurements[beacon_name]) - 1)

                        # Overwrite the chosen measurement with the most recent measurement.
                        self.__beacon_measurements[beacon_name][idx] = self.__beacon_measurements[beacon_name][-1]

                        # Overwrite the most recent measurement with the new measurement.
                        self.__beacon_measurements[beacon_name][-1] = beacon_measurement

                    # Otherwise, simply add the new measurement to the list.
                    else:
                        self.__beacon_measurements[beacon_name].append(beacon_measurement)

                    # Mark the beacon as one for which new measurements have been added since localisation was
                    # last run. This will cause localisation to be re-run for the beacon when there's time.
                    self.__dirty_beacons.add(beacon_name)

    def get_beacon_measurements(self) -> Dict[str, List[Tuple[np.ndarray, float]]]:
        """
        TODO

        :return:    TODO
        """
        with self.__lock:
            return copy.deepcopy(self.__beacon_measurements)

    def get_beacons(self) -> Dict[str, Beacon]:
        """
        Get all of the beacons that are known by the localiser.

        .. note::
            We copy the beacons whose positions might subsequently change so that callers can use the returned
            beacons without having to worry about this.

        :return:    All of the beacons that are known by the localiser.
        """
        with self.__lock:
            return {**copy.deepcopy(self.__localised_beacons), **self.get_fake_beacons()}

    def get_fake_beacons(self) -> Dict[str, Beacon]:
        """
        Get the fake beacons that are known by the localiser.

        :return:    The fake beacons that are known by the localiser.
        """
        return self.__fake_beacons

    def set_fake_beacon(self, beacon_name: str, beacon: Optional[Beacon]) -> None:
        """
        Set a fake beacon at a particular position in the scene, or clear an existing one.

        .. note::
            There can be at most one beacon with any given name, so reusing a name overwrites the existing beacon.
        .. note::
            To clear a beacon, it suffices to call set_fake_beacon(name, None).

        :param beacon_name: The name of the beacon.
        :param beacon:      Either the beacon (which contains its own position), or None (to clear the beacon with
                            the given name).
        """
        # Set, overwrite, or clear the fake beacon as requested.
        if beacon is not None:
            self.__fake_beacons[beacon_name] = beacon
        elif beacon_name in self.__fake_beacons:
            del self.__fake_beacons[beacon_name]

        with self.__lock:
            # Clear any existing measurements associated with this beacon name.
            if beacon_name in self.__beacon_measurements:
                del self.__beacon_measurements[beacon_name]

            # Also clear the associated localised beacon, if it exists.
            if f"L_{beacon_name}" in self.__localised_beacons:
                del self.__localised_beacons[f"L_{beacon_name}"]

    def terminate(self) -> None:
        """Tell the localiser to terminate."""
        if not self.__should_terminate.is_set():
            self.__should_terminate.set()

            # Wait for the localisation thread to terminate.
            self.__localisation_thread.join()

    # PRIVATE METHODS

    def __run_localisation(self) -> None:
        """Run the beacon localisation thread."""
        # Until the localiser should terminate:
        while not self.__should_terminate.is_set():
            # Make a copy of the shared variables so that we only need to hold the lock very briefly.
            with self.__lock:
                beacon_measurements: Dict[str, List[Tuple[np.ndarray, float]]] = copy.deepcopy(
                    self.__beacon_measurements
                )
                dirty_beacons: Set[str] = self.__dirty_beacons.copy()
                self.__dirty_beacons.clear()

            # For each beacon for which new measurements have been added since the last localisation attempt:
            for beacon_name in dirty_beacons:
                # Get the measurements for the beacon.
                measurements_for_beacon: Optional[List[Tuple[np.ndarray, float]]] = beacon_measurements.get(beacon_name)
                if measurements_for_beacon is None:
                    continue

                # Try to localise the beacon.
                beacon_pos: Optional[np.ndarray] = BeaconLocaliser.try_localise_beacon(measurements_for_beacon)

                # If that succeeded:
                if beacon_pos is not None:
                    with self.__lock:
                        # Set or overwrite the localised version of the beacon.
                        max_range: float = max(measurements_for_beacon, key=operator.itemgetter(1))[1]
                        self.__localised_beacons[f"L_{beacon_name}"] = Beacon(
                            beacon_pos, max_range, Beacon.BT_LOCALISED
                        )

                    # Optionally output some debugging information.
                    if self.__debug:
                        print(f"Beacon {beacon_name} localised at {beacon_pos}!", self.__fake_beacons["Foo"].position)

            # Wait momentarily to avoid a spin loop.
            time.sleep(0.01)

    # PRIVATE STATIC METHODS

    @staticmethod
    def __mean_square_error(beacon_pos: np.ndarray, beacon_measurements: List[Tuple[np.ndarray, float]]) -> float:
        """
        TODO

        :param beacon_pos:          TODO
        :param beacon_measurements: TODO
        :return:                    TODO
        """
        # TODO
        mse: float = 0.0
        for receiver_pos, measured_distance in beacon_measurements:
            calculated_distance: float = np.linalg.norm(receiver_pos - beacon_pos)
            mse += math.pow(calculated_distance - measured_distance, 2.0)

        return mse / len(beacon_measurements)
