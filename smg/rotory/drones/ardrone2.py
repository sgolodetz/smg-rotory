import av
import numpy as np
import socket
import struct
import sys
import threading
import time

from collections import namedtuple
from distutils.util import strtobool
from typing import Dict, List, Optional, Tuple

from smg.rotory.net.udp_link import UDPLink
from smg.rotory.util.bits_util import BitsUtil


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
                 cmd_endpoint: Tuple[str, int] = ("192.168.1.1", 5556),
                 control_endpoint: Tuple[str, int] = ("192.168.1.1", 5559),
                 local_ip: str = "192.168.1.2",
                 navdata_endpoint: Tuple[str, int] = ("192.168.1.1", 5554),
                 print_commands: bool = True,
                 print_control_messages: bool = True,
                 print_navdata_messages: bool = True,
                 video_endpoint: Tuple[str, int] = ("192.168.1.1", 5555)):
        """
        Construct an ARDrone2 object, which provides a convenient interface to control a Parrot AR Drone 2.

        :param cmd_endpoint:            The remote endpoint (IP address and port) to which to send AT commands.
        :param control_endpoint:        The remote endpoint (IP address and port) from which to receive critical data.
        :param navdata_endpoint:        The remote endpoint (IP address and port) from which to receive navigation data.
        :param print_commands:          Whether or not to print commands that are sent.
        :param print_control_messages:  Whether or not to print control messages.
        :param print_navdata_messages:  Whether or not to print navdata messages.
        :param video_endpoint:          The remote endpoint (IP address and port) from which to receive video.
        """
        self.__drone_state: str = ""
        self.__frame_is_pending: bool = False
        self.__front_buffer: np.ndarray = np.zeros((360, 640), dtype=np.uint8)
        self.__navdata_is_available: bool = False
        self.__navdata_options: Dict[str, bytes] = {}
        self.__pave_header_fmt: str = "4sBBHIHHHHIIBBBBIIHBBBB2sI12s"
        self.__pave_header_size: int = struct.calcsize(self.__pave_header_fmt)
        self.__print_commands: bool = print_commands
        self.__print_control_messages: bool = print_control_messages
        self.__print_navdata_messages: bool = print_navdata_messages
        self.__sequence_number: int = 1
        self.__should_terminate: bool = False

        # Set up the locks and conditions.
        self.__cmd_lock = threading.Lock()
        self.__navdata_lock = threading.Lock()
        self.__video_lock = threading.Lock()

        self.__navdata_ready = threading.Condition(self.__navdata_lock)
        self.__no_pending_frame = threading.Condition(self.__video_lock)

        # Set up the TCP connections.
        self.__control_socket: socket.SocketType = ARDrone2.__make_tcp_socket(control_endpoint)
        self.__video_socket: socket.SocketType = ARDrone2.__make_tcp_socket(video_endpoint)

        # Set up the UDP links.
        self.__cmd_link = UDPLink((local_ip, 5556), cmd_endpoint)
        self.__navdata_link = UDPLink((local_ip, 5554), navdata_endpoint)

        # Start the threads.
        self.__control_thread = threading.Thread(target=self.__process_control_messages)
        self.__control_thread.start()
        self.__heartbeat_thread = threading.Thread(target=self.__process_heartbeats)
        self.__heartbeat_thread.start()
        self.__navdata_thread = threading.Thread(target=self.__process_navdata_messages)
        self.__navdata_thread.start()
        self.__video_thread = threading.Thread(target=self.__process_video_messages)
        self.__video_thread.start()

        # Request the current drone configuration.
        self.__send_command("CTRL", 4, 0)

        # Ask the drone to start sending full navdata messages.
        self.__send_command("CONFIG", "general:navdata_demo", "FALSE")

        # Reset the drone control mode.
        self.__send_command("CTRL", 5, 0)
        self.__send_command("CTRL", 0, 0)

        # Trim the drone prior to takeoff.
        self.__send_command("FTRIM")

        # Wait for the navdata to become available.
        with self.__navdata_lock:
            while not self.__navdata_is_available and not self.__should_terminate:
                self.__navdata_ready.wait(0.1)

    # SPECIAL METHODS

    def __enter__(self):
        """No-op (needed to allow the drone object's lifetime to be managed by a with statement)."""
        return self

    def __exit__(self, exception_type, exception_value, traceback):
        """Destroy the drone object at the end of the with statement that's used to manage its lifetime."""
        self.__should_terminate = True

        # Wait for all of the threads to terminate.
        self.__control_thread.join()
        self.__heartbeat_thread.join()
        self.__navdata_thread.join()
        self.__video_thread.join()

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

    def get_navdata_option(self, name: str) -> Optional[bytes]:
        """
        TODO

        :param name:    TODO
        :return:        TODO
        """
        with self.__navdata_lock:
            return self.__navdata_options.get(name)

    def get_time(self) -> Optional[Tuple[int, int]]:
        """
        TODO

        :return:    TODO
        """
        time_option: Optional[bytes] = self.get_navdata_option("time")
        if time_option is not None and len(time_option) == 4:
            time_bits: str = f"{struct.unpack('i', time_option)[0]:032b}"
            secs: int = BitsUtil.convert_hilo_bit_string_to_int32(time_bits[:11])
            microsecs: int = BitsUtil.convert_hilo_bit_string_to_int32(time_bits[11:])
            return secs, microsecs
        else:
            return None

    def land(self):
        """Tell the drone to land."""
        # Make the argument to the REF command that's used to tell the drone to land.
        bits: List[str] = list("0" * 32)
        bits[18] = bits[20] = bits[22] = bits[24] = bits[28] = "1"
        arg: int = BitsUtil.convert_lohi_bit_string_to_int32("".join(bits))

        # Get the initial flying state.
        flying: bool = self.__get_drone_state_bit(0)

        # Until the drone has landed (or program termination has been requested):
        while flying and not self.__should_terminate:
            # Send the landing command.
            self.__send_command("REF", arg)

            # Sleep for 30 milliseconds.
            time.sleep(0.03)

            # Check whether the drone's flying state has changed to reflect the landing.
            flying = self.__get_drone_state_bit(0)

    def takeoff(self):
        """Tell the drone to take off."""
        # Make the argument to the REF command that's used to tell the drone to take off.
        bits: List[str] = list("0" * 32)
        bits[9] = bits[18] = bits[20] = bits[22] = bits[24] = bits[28] = "1"
        arg: int = BitsUtil.convert_lohi_bit_string_to_int32("".join(bits))

        # Get the initial flying state.
        flying: bool = self.__get_drone_state_bit(0)

        # Until the drone has taken off (or program termination has been requested):
        while not flying and not self.__should_terminate:
            # Send the takeoff command.
            self.__send_command("REF", arg)

            # Sleep for 30 milliseconds.
            time.sleep(0.03)

            # Check whether the drone's flying state has changed to reflect the takeoff.
            flying = self.__get_drone_state_bit(0)

    # PRIVATE METHODS

    def __get_drone_state_bit(self, bit: int) -> bool:
        """
        Get the specified bit from the most recent drone state retrieved from the navdata.

        .. note::
            The drone state is stored as a 32-bit binary string, with the low bits first.
        .. note::
             The detailed meaning of each bit can be found in ARDrone_SDK_2_0_1/ARDroneLib/Soft/Common/config.h.

             0: FLY MASK; 1: VIDEO MASK; 2: VISION MASK; 3: CONTROL ALGO;
             4: ALTITUDE CONTROL ALGO; 5: USER feedback; 6: Control command ACK; 7: Camera enable;
             8: Travelling enable; 9: USB key; 10: Navdata demo; 11: Navdata bootstrap;
             12: Motors status; 13: Communication Lost; 14: <Not used>; 15: VBat low;
             16: User Emergency Landing; 17: Timer elapsed; 18: Magnetometer calibration state; 19: Angles;
             20: WIND MASK; 21: Ultrasonic sensor; 22: Cutout system detection; 23: PIC Version number OK;
             24: ATCodec thread ON; 25: Navdata thread ON; 26: Video thread ON; 27: Acquisition thread ON;
             28: CTRL watchdog; 29: ADC Watchdog; 30: Communication Watchdog; 31: Emergency landing

        :param bit: The index of the bit to get (must be between 0 and 31, inclusive).
        :return:    The specified bit from the drone state.
        """
        with self.__navdata_lock:
            if 0 <= bit < 32:
                return bool(strtobool(self.__drone_state[bit]))
            else:
                raise ValueError(f"Cannot get bit #{bit} of the 32-bit drone state")

    def __process_control_messages(self) -> None:
        """Process control messages sent by the drone."""
        # While the drone should not terminate.
        while not self.__should_terminate:
            # Attempt to receive a control message from the drone.
            try:
                control_message = self.__control_socket.recv(32768)
            except socket.timeout:
                control_message = b"timeout"

            # If this unblocks because the drone should terminate, exit.
            if self.__should_terminate:
                return

            # Print the control message if desired.
            if self.__print_control_messages:
                print(f"Control Message ({len(control_message)}): {control_message}")

    def __process_heartbeats(self) -> None:
        """TODO"""
        while not self.__should_terminate:
            # Sleep for 30 milliseconds.
            time.sleep(0.03)

            # Send a COMWDG command to the drone to keep it awake.
            self.__send_command("COMWDG")

    def __process_navdata_messages(self) -> None:
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

            # Print the navdata message that was sent, if desired.
            if self.__print_navdata_messages:
                print(f"NavData Message: {navdata_message}")

            # Try to process the navdata message. If anything goes wrong, simply print a warning and skip it.
            if len(navdata_message) >= 16:
                header, drone_state, sequence_number, vision_flag = struct.unpack("iiii", navdata_message[:16])
                if header == 0x55667788:
                    with self.__navdata_lock:
                        # Convert the 32-bit drone state to a 32-bit binary string, with the low bits first.
                        self.__drone_state = BitsUtil.convert_int32_to_lohi_bit_string(drone_state)

                        # Unpack the navdata options.
                        self.__navdata_options = ARDrone2.unpack_navdata_options(navdata_message)

                        # Signal that the navdata is available.
                        self.__navdata_is_available = True
                        self.__navdata_ready.notify()
                else:
                    print("Warning: Incoming navdata message had a bad header; skipping")
            else:
                print("Warning: Incoming navdata message was too short; skipping")

    # noinspection PyUnresolvedReferences
    def __process_video_messages(self) -> None:
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

    def __send_command(self, name: str, *args) -> None:
        """
        Send the specified command to the drone.

        .. note::
            It is essential that the arguments passed in to this function have the right types, since the types
            are used internally to format the arguments ready for sending across to the drone. For example,
            it would be really bad to pass in a whole number as an int rather than a float for an argument
            that's supposed to be floating-point, since ints and floats are processed differently prior to
            being sent across to the drone (see https://jpchanson.github.io/ARdrone/ParrotDevGuide.pdf).

        :param name:    The name of the command to send.
        :param args:    The arguments to the command.
        """
        with self.__cmd_lock:
            # Make the command and send it to the drone.
            cmd: bytes = ARDrone2.__make_at_command(name, self.__sequence_number, *args)
            self.__cmd_link.socket.sendto(cmd, self.__cmd_link.remote_endpoint)

            # Update the sequence number ready for the next command.
            self.__sequence_number += 1

            # Print the command that was sent, if desired.
            if self.__print_commands:
                print(f"Sent Command: {cmd}")

    # PRIVATE STATIC METHODS

    @staticmethod
    def __get_navdata_option_name(tag: int) -> Optional[str]:
        """
        TODO

        .. note::
            The various tags are listed in ARDrone_SDK_2_0_1/ARDroneLib/Soft/Common/navdata_keys.h, and the
            corresponding structs are listed in ARDrone_SDK_2_0_1/ARDroneLib/Soft/Common/navdata_common.h.
            Note that the various macros in navdata_keys.h get redefined in navdata.c (specifically, in the
            ardrone_navdata_unpack_all function), so that's why they're confusingly defined as empty in the
            header file. For clarity here, the various options are:

            0: NAVDATA_DEMO_TAG (size 148)
            1: NAVDATA_TIME_TAG (size 8)
            2: NAVDATA_RAW_MEASURES_TAG (size 52)
            3: NAVDATA_PHYS_MEASURES_TAG (size 46)
            4: NAVDATA_GYROS_OFFSETS_TAG (size 16)
            5: NAVDATA_EULER_ANGLES_TAG (size 12)
            6: NAVDATA_REFERENCES_TAG (size 88)
            7: NAVDATA_TRIMS_TAG (size 16)
            8: NAVDATA_RC_REFERENCES_TAG (size 24)
            NAVDATA_PWM_TAG [SIZE IS 76, SEEMS WRONG]
            10: NAVDATA_ALTITUDE_TAG (size 56) [SIZE SEEMS RIGHT, BUT COULD BE CONFUSED WITH NAVDATA_WIND_TAG]
            11: NAVDATA_VISION_RAW_TAG (size 16)
            12: NAVDATA_VISION_OF_TAG (size 44)
            13: NAVDATA_VISION_TAG (size 92)
            14: NAVDATA_VISION_PERF_TAG (size 108)
            15: NAVDATA_TRACKERS_SEND_TAG (size 364)
            16: NAVDATA_VISION_DETECT_TAG (size 328)
            17: NAVDATA_WATCHDOG_TAG (size 8)
            18: NAVDATA_ADC_DATA_FRAME_TAG (size 40)
            19: NAVDATA_VIDEO_STREAM_TAG (size 65)
            20: NAVDATA_GAMES_TAG (size 12)
            21: NAVDATA_PRESSURE_RAW_TAG (size 18)
            NAVDATA_MAGNETO_TAG [SIZE IS 75, SEEMS WRONG]
            NAVDATA_WIND_TAG [SIZE IS 56, SEEMS WRONG, COULD BE CONFUSED WITH #10]
            24: NAVDATA_KALMAN_PRESSURE_TAG (size 72)
            25: NAVDATA_HDVIDEO_STREAM_TAG (size 32)
            26: NAVDATA_WIFI_TAG (size 8)
            NAVDATA_ZIMMU_3000_TAG [SIZE IS 12, SEEMS WRONG, COULD BE CONFUSED WITH #20]
            -1: NAVDATA_CKS_TAG (size 8)

        :param tag: TODO
        :return:    TODO
        """
        return {
            0: "demo",
            1: "time",
            2: "raw_measures",
            3: "phys_measures",
            4: "gyros_offsets",
            5: "euler_angles",
            6: "references",
            7: "trims",
            8: "rc_references",
            # ? : "pwm",
            10: "altitude",  # TODO: Make sure that this is correct, there's another option with the same size.
            11: "vision_raw",
            12: "vision_of",
            13: "vision",
            14: "vision_perf",
            15: "trackers_send",
            16: "vision_detect",
            17: "watchdog",
            18: "adc_data_frame",
            19: "video_stream",
            20: "games",
            21: "pressure_raw",
            # ? : "magneto",
            # ? : "wind",
            24: "kalman_pressure",
            25: "hdvideo_stream",
            26: "wifi",
            # ? : "zimmu_3000",
            -1: "cks"
        }.get(tag)

    @staticmethod
    def __get_at_arg_count(name: str) -> int:
        """
        Get the number of arguments accepted by an AT command with the specified name.

        :param name:    The name of the AT command.
        :return:        The number of arguments that the AT command accepts.
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
    def __make_at_command(name: str, sequence_number: int, *args) -> bytes:
        """
        Make an AT command to send to the drone.

        :param name:                The name of the command.
        :param sequence_number:     The sequence number to embed in the command.
        :param args:                The arguments to the command.
        :return:                    The command.
        """
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

    @staticmethod
    def unpack_navdata_options(navdata_message: bytes) -> Dict[str, bytes]:
        """
        TODO

        :param navdata_message:     TODO
        """
        # FIXME: Make this private.

        navdata_options: Dict[str, bytes] = {}

        # TODO
        offset: int = 16

        # TODO
        while offset < len(navdata_message) - 4:
            # TODO
            tag, size = struct.unpack_from("hh", navdata_message, offset)

            # TODO
            if offset + size <= len(navdata_message):
                # TODO
                name: Optional[str] = ARDrone2.__get_navdata_option_name(tag)

                # TODO
                if name is not None:
                    navdata_options[name] = navdata_message[offset+4:offset+size]

            # TODO
            offset += size

        return navdata_options
