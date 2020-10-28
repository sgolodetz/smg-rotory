import socket


class UDPLink:
    """A bidirectional UDP link."""

    # CONSTRUCTOR

    def __init__(self, local_endpoint, remote_endpoint):
        """
        Construct a bidirectional UDP link.

        :param local_endpoint:      The local endpoint.
        :param remote_endpoint:     The remote endpoint.
        """
        self.local_endpoint = local_endpoint
        self.remote_endpoint = remote_endpoint
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.socket.bind(local_endpoint)
        self.socket.settimeout(10)
