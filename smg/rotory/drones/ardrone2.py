import av
import numpy as np
import socket
import struct
import sys
import threading

from collections import namedtuple
from typing import Dict, List, Tuple

from smg.rotory.net.udp_link import UDPLink


class ARDrone2:
    """An interface that can be used to control a Parrot AR Drone 2."""

    # NESTED TYPES

    PaVEHeader = namedtuple(
        'PaVEHeader', [
            'signature', 'version', 'video_codec', 'header_size', 'payload_size',
            'encoded_stream_width', 'encoded_stream_height', 'display_width', 'display_height',
            'frame_number', 'timestamp', 'total_chunks', 'chunk_index', 'frame_type', 'control',
            'stream_byte_position_lw', 'stream_byte_position_uw', 'stream_id', 'total_slices', 'slice_index',
            'header1_size', 'header2_size', 'reserved2', 'advertised_size', 'reserved3'
        ]
    )

    # CONSTRUCTORS

    def __init__(self, *,
                 local_ip: str = "192.168.1.2",
                 navdata_endpoint: Tuple[str, int] = ("192.168.1.1", 5554),
                 video_endpoint: Tuple[str, int] = ("192.168.1.1", 5555)):
        """
        Construct an ARDrone2 object, which provides a convenient interface to control a Parrot AR Drone 2.

        :param navdata_endpoint:    The remote endpoint (IP address and port) from which to receive navigation data.
        :param video_endpoint:      The remote endpoint (IP address and port) from which to receive video.
        """
        self.__cmd_sequence_number: int = 1
        self.__frame_is_pending: bool = False
        self.__front_buffer: np.ndarray = np.zeros((360, 640), dtype=np.uint8)
        self.__pave_header_fmt: str = "4sBBHIHHHHIIBBBBIIHBBBB2sI12s"
        self.__pave_header_size: int = struct.calcsize(self.__pave_header_fmt)
        self.__should_terminate: bool = False

        # Set up the locks and conditions.
        self.__cmd_lock = threading.Lock()
        self.__video_lock = threading.Lock()

        self.__no_pending_frame = threading.Condition(self.__video_lock)

        # Set up the TCP connections.
        self.__video_socket: socket.SocketType = ARDrone2.__make_tcp_socket(video_endpoint)

        # Set up the UDP links.
        self.__navdata_link = UDPLink((local_ip, 5554), navdata_endpoint)

        # Start the threads.
        self.__navdata_thread = threading.Thread(target=self.__process_navdata_messages)
        self.__navdata_thread.start()
        self.__video_thread = threading.Thread(target=self.__process_video_messages)
        self.__video_thread.start()

    # SPECIAL METHODS

    def __enter__(self):
        """No-op (needed to allow the drone object's lifetime to be managed by a with statement)."""
        return self

    def __exit__(self, exception_type, exception_value, traceback):
        """Destroy the drone object at the end of the with statement that's used to manage its lifetime."""
        self.__should_terminate = True

        # Artificially wake any waiting threads so that they can terminate.
        # TODO: Check whether this is really necessary in Python, given that we're using condition timeouts.

        # Wait for all of the threads to terminate.
        self.__navdata_thread.join()
        self.__video_thread.join()

    # PUBLIC STATIC METHODS

    @staticmethod
    def make_at_command(name: str, sequence_number: int, *args) -> bytes:
        """
        TODO

        :param name:                TODO
        :param sequence_number:     TODO
        :param args:                TODO
        :return:                    TODO
        """
        # TODO: Make this method private once I've finished checking it.
        modified_args: List[str] = []

        for arg in args:
            t = type(arg)
            if t is float:
                modified_args.append(str(struct.unpack("i", struct.pack("f", arg))[0]))
            elif t is int:
                modified_args.append(str(arg))
            elif t is str:
                modified_args.append('"' + arg + '"')

        fmt: str = f"AT*{name}=" + ("{}," * ARDrone2.__get_at_arg_count(name))[:-1] + "\r"
        return bytes(fmt.format(sequence_number, *modified_args), "utf-8")

    # PUBLIC METHODS

    def get_image(self) -> np.ndarray:
        """
        Get the most recent image received from the Tello.

        :return:    The most recent image received from the Tello.
        """
        with self.__video_lock:
            while self.__frame_is_pending and not self.__should_terminate:
                self.__no_pending_frame.wait(0.1)

            return self.__front_buffer.copy()

    # PRIVATE METHODS

    def __process_navdata_messages(self):
        """Process navdata messages sent by the drone."""
        # Ask the drone to start sending navdata messages.
        # See: https://github.com/afq984/pyardrone/blob/master/pyardrone/navdata/__init__.py
        self.__navdata_link.socket.sendto(b"\x01\x00\x00\x00", self.__navdata_link.remote_endpoint)

        # While the drone should not terminate:
        while not self.__should_terminate:
            # Attempt to receive a navdata message from the drone.
            try:
                navdata_message, source = self.__navdata_link.socket.recvfrom(4096)
            except socket.timeout:
                navdata_message, source = b"timeout", self.__navdata_link.local_endpoint

            # If this unblocks because the drone should terminate, exit.
            if self.__should_terminate:
                return

            print(f"NavData Message: {navdata_message}")

    # noinspection PyUnresolvedReferences
    def __process_video_messages(self):
        """Process video messages sent by the drone."""
        # Tell PyAV not to print out any non-fatal logging messages (it's quite verbose otherwise).
        av.logging.set_level(av.logging.FATAL)

        # Create the H.264 codec and the context we're going to use to decode the video stream.
        codec = av.codec.Codec("h264")
        context = av.codec.CodecContext.create(codec)
        context.pix_fmt = "yuv420p"
        context.open()

        # Initialise the frame accumulation buffer and the back buffer.
        accum_buffer = b""
        back_buffer = None

        # Initialise the expected size of the next frame packet (it needs to be bigger than the size of
        # any incoming non-PaVE message so that we don't try to decode a packet until we're ready).
        frame_packet_size: int = sys.maxsize

        # While the drone should not terminate:
        while not self.__should_terminate:
            # Attempt to receive a video message from the drone.
            try:
                video_message = self.__video_socket.recv(4096)
            except socket.timeout:
                video_message = b"timeout"

            # If this unblocks because the drone should terminate, exit.
            if self.__should_terminate:
                return

            # If the video message has a PaVE header, extract the expected frame packet size from it, and overwrite
            # the accumulation buffer with the contents of the message. Otherwise, append the message to the buffer.
            if len(video_message) >= self.__pave_header_size and video_message.startswith(b"PaVE"):
                # Unpack the PaVE header.
                pave_header_fields = struct.unpack(self.__pave_header_fmt, video_message[:self.__pave_header_size])
                pave_header: ARDrone2.PaVEHeader = ARDrone2.PaVEHeader._make(pave_header_fields)

                # Calculate the expected frame packet size.
                frame_packet_size: int = pave_header.header_size + pave_header.payload_size

                # Overwrite the accumulation buffer with the contents of the video message.
                accum_buffer = video_message
            else:
                # Append the video message to the accumulation buffer.
                accum_buffer += video_message

            # If we reached the end of the frame (see ARDroneLib's video_stage_tcp.c, L345):
            if len(accum_buffer) >= frame_packet_size:
                # Make a packet from the contents of the accumulation buffer.
                packet = av.packet.Packet(accum_buffer)

                try:
                    # Try to decode the packet.
                    frames = context.decode(packet)

                    # If we successfully decoded any frames, copy one of them into the back buffer,
                    # and make it clear that we're going to want the video lock.
                    if len(frames) > 0:
                        back_buffer = frames[0].to_ndarray(format='bgr24')
                        self.__frame_is_pending = True
                except av.AVError:
                    # If the packet could not be successfully decoded, just ignore it and carry on.
                    pass

            # If the back buffer has been updated since the last time it was copied to the front
            # buffer, try to copy it now. If that's not possible right now (e.g. because another
            # thread is busy reading the front buffer), just carry on.
            if back_buffer is not None:
                acquired = self.__video_lock.acquire(blocking=False)
                if acquired:
                    self.__front_buffer = back_buffer
                    back_buffer = None
                    self.__frame_is_pending = False
                    self.__no_pending_frame.notify_all()
                    self.__video_lock.release()

    # PRIVATE STATIC METHODS

    @staticmethod
    def __get_at_arg_count(name: str) -> int:
        """
        TODO

        :param name:    TODO
        :return:        TODO
        """
        arg_counts: Dict[str, int] = {
            "CALIB": 2,
            "COMWDG": 1,
            "CONFIG": 3,
            "CONFIG_IDS": 4,
            "CTRL": 3,
            "FTRIM": 1,
            "PCMD": 6,
            "PCMD_MAG": 8,
            "REF": 2
        }

        result: int = arg_counts.get(name)

        if result is not None:
            return result
        else:
            raise ValueError(f"Unknown command: {name}")

    @staticmethod
    def __make_tcp_socket(endpoint: Tuple[str, int], *, timeout: int = 10) -> socket.SocketType:
        """
        Make a TCP socket, connect to the specified endpoint, and set an appropriate timeout.

        :param endpoint:    The endpoint to which to connect.
        :param timeout:     The timeout for the socket.
        :return:            The socket.
        """
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.connect(endpoint)
        sock.settimeout(timeout)
        return sock
