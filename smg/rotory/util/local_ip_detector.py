from netifaces import AF_INET, ifaddresses, interfaces
from typing import List, Optional


class LocalIPDetector:
    """Utility functions for determining local IP addresses on this host."""

    # PUBLIC STATIC METHODS

    @staticmethod
    def get_all_ips() -> List[str]:
        """
        Get a list of all of this host's IP addresses.

        :return:    A list of all of this host's IP addresses.
        """
        result = []  # type: List[str]
        for interface in interfaces():
            for link in ifaddresses(interface).get(AF_INET, []):
                result.append(link["addr"])
        return result

    @staticmethod
    def get_ip_starting_with(partial_ip: str) -> Optional[str]:
        """
        Get any IP address that this host has that starts with the specified partial IP address.

        .. note::
            If this host has multiple IP addresses that start with the partial IP address passed in,
            only one of them will be returned. Notably, this will at least be the same one each time.

        :param partial_ip:  A partial IP address, e.g. "192.168.10".
        :return:            Any IP address that this host has that starts with the specified partial IP address,
                            if possible, or None otherwise.
        """
        for ip in LocalIPDetector.get_all_ips():
            if ip.startswith(partial_ip):
                return ip
        return None
