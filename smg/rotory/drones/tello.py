import av
import numpy as np
import socket
import threading
import time

from typing import Dict, Mapping, Optional, Tuple

from smg.rotory.net.udp_link import UDPLink


class Tello:
    """An interface that can be used to control a DJI Tello drone."""

    # CONSTRUCTOR

    def __init__(self, *,
                 local_ip: str = "192.168.10.2",
                 remote_endpoint: Tuple[str, int] = ("192.168.10.1", 8889),
                 print_commands: bool = False,
                 print_responses: bool = False,
                 print_state_messages: bool = False):
        """
        Construct a Tello object, which provides a convenient interface to control a DJI Tello drone.

        :param local_ip:                The IP address of the local machine.
        :param remote_endpoint:         The remote endpoint (IP address and port) to which to send commands.
        :param print_commands:          Whether or not to print commands that are sent.
        :param print_responses:         Whether or not to print command responses.
        :param print_state_messages:    Whether or not to print state messages.
        """
        self.__frame_is_pending: bool = False
        self.__front_buffer: np.ndarray = np.zeros((720, 960, 3), dtype=np.uint8)
        self.__print_commands: bool = print_commands
        self.__print_responses: bool = print_responses
        self.__print_state_messages: bool = print_state_messages
        self.__rc_forward = 0
        self.__rc_right = 0
        self.__rc_up = 0
        self.__rc_yaw = 0
        self.__response_is_pending: bool = False
        self.__should_terminate: bool = False
        self.__state_map: dict = {}

        # Set up the locks and conditions.
        self.__cmd_lock = threading.Lock()
        self.__state_lock = threading.Lock()
        self.__video_lock = threading.Lock()

        self.__no_pending_frame = threading.Condition(self.__video_lock)
        self.__no_pending_response = threading.Condition(self.__cmd_lock)
        self.__pending_response = threading.Condition(self.__cmd_lock)

        # Set up the UDP links.
        self.__cmd_link = UDPLink((local_ip, 8888), remote_endpoint)
        self.__state_link = UDPLink((local_ip, 8890), remote_endpoint)
        self.__video_link = UDPLink((local_ip, 11111), remote_endpoint)

        # Start the threads.
        self.__heartbeat_thread = threading.Thread(target=self.__process_heartbeats)
        self.__heartbeat_thread.start()
        self.__response_thread = threading.Thread(target=self.__process_command_responses)
        self.__response_thread.start()
        self.__state_thread = threading.Thread(target=self.__process_state_messages)
        self.__state_thread.start()
        self.__video_thread = threading.Thread(target=self.__process_video_messages)
        self.__video_thread.start()

        # Tell the drone to switch into SDK mode and start streaming video.
        self.__send_command("command", expect_response=True)
        self.__send_command("streamon", expect_response=True)

    # SPECIAL METHODS

    def __enter__(self):
        """No-op (needed to allow the drone object's lifetime to be managed by a with statement)."""
        return self

    def __exit__(self, exception_type, exception_value, traceback):
        """Destroy the drone object at the end of the with statement that's used to manage its lifetime."""
        self.__should_terminate = True

        # Artificially wake any waiting threads so that they can terminate.
        # TODO: Check whether this is really necessary in Python, given that we're using condition timeouts.
        with self.__cmd_lock:
            self.__no_pending_response.notify_all()
            self.__pending_response.notify_all()

        # Wait for all of the threads to terminate.
        self.__heartbeat_thread.join()
        self.__response_thread.join()
        self.__state_thread.join()
        self.__video_thread.join()

    # PUBLIC METHODS

    def get_battery_level(self) -> Optional[int]:
        """
        Try to get the most recently received value of the remaining battery %.

        :return:    The most recently received value of the remaining battery %, if available, or None otherwise.
        """
        with self.__state_lock:
            bat: Optional[str] = self.__state_map.get("bat")
            return int(bat) if bat is not None else None

    def get_image(self) -> np.ndarray:
        """
        Get the most recent image received from the drone.

        :return:    The most recent image received from the drone.
        """
        with self.__video_lock:
            while self.__frame_is_pending and not self.__should_terminate:
                self.__no_pending_frame.wait(0.1)

            return self.__front_buffer.copy()

    def get_state_map(self) -> Dict[str, str]:
        """
        Get the most recent state map received from the drone.

        :return:    The most recent state map received from the drone.
        """
        with self.__state_lock:
            return self.__state_map.copy()

    def land(self):
        """Tell the drone to land."""
        self.__send_command("land", expect_response=True)

    def move_forward(self, rate: float):
        """
        Tell the drone to move forward at the specified rate.

        This can also be used to move backwards (by specifying a negative rate).

        :param rate:     The rate at which the drone should move forward (in [-1,1]).
        """
        self.__rc_forward = Tello.__rate_to_control_value(rate)

    def move_right(self, rate: float):
        """
        Tell the drone to move to the right at the specified rate.

        This can also be used to move to the left (by specifying a negative rate).

        :param rate:    The rate at which the drone should move to the right (in [-1,1]).
        """
        self.__rc_right = Tello.__rate_to_control_value(rate)

    def move_up(self, rate: float):
        """
        Tell the drone to move up at the specified rate.

        This can also be used to move down (by specifying a negative rate).

        :param rate:    The rate at which the drone should move up (in [-1,1]).
        """
        self.__rc_up = Tello.__rate_to_control_value(rate)

    def stop(self):
        """Tell the drone to stop in mid-air."""
        self.__rc_right = 0
        self.__rc_forward = 0
        self.__rc_yaw = 0

    def takeoff(self):
        """Tell the drone to take off."""
        self.__send_command("takeoff", expect_response=True)

    def turn(self, rate: float):
        """
        Tell the drone to turn at the specified rate.

        :param rate:    The rate at which the drone should turn (in [-1,1]).
        """
        self.__rc_yaw = Tello.__rate_to_control_value(rate)

    # PRIVATE METHODS

    def __process_command_responses(self) -> None:
        """Process command responses sent by the drone."""
        with self.__cmd_lock:
            # While the drone should not terminate:
            while not self.__should_terminate:
                # Wait until a command has been sent to which a response is expected.
                while not self.__response_is_pending:
                    self.__pending_response.wait(0.1)

                    # If the thread has been artificially woken to allow it to terminate, exit.
                    if self.__should_terminate:
                        return

                # Attempt to receive the response to the command from the drone.
                try:
                    response, source = self.__cmd_link.socket.recvfrom(2048)
                except socket.timeout:
                    response, source = b"timeout", self.__cmd_link.local_endpoint

                # We should only get here if the response is non-empty.
                assert(len(response) > 0)

                # Print the response if desired.
                if self.__print_responses:
                    print("Response ({}): {}".format(len(response), response))

                # Handle any errors that occur.
                # TODO

                # Allow new commands to be sent.
                self.__response_is_pending = False
                self.__no_pending_response.notify()

    def __process_heartbeats(self) -> None:
        """Send regular flight control ("rc") messages to keep the drone awake."""
        # Send a flight control ("rc") command every 20ms to ensure that the drone stays awake.
        # Note #1: This is also how we control the drone's flight - it serves both purposes.
        # Note #2: The drone doesn't respond to "rc" commands, so we don't wait for a response.
        while not self.__should_terminate:
            # Sleep for 20 milliseconds.
            time.sleep(0.02)

            # Construct the "rc" command to send to the drone.
            cmd = "rc {} {} {} {}".format(self.__rc_right, self.__rc_forward, self.__rc_up, self.__rc_yaw)

            # Send the command.
            self.__send_command(cmd, expect_response=False)

    def __process_state_messages(self) -> None:
        """Process state messages sent by the drone."""
        # While the drone should not terminate:
        while not self.__should_terminate:
            # Sleep for 100 milliseconds.
            time.sleep(0.1)

            # Attempt to receive a state message from the drone.
            try:
                state_message, source = self.__state_link.socket.recvfrom(2048)
            except socket.timeout:
                state_message, source = b"timeout", self.__state_link.local_endpoint

            # If this unblocks because the drone should terminate, exit.
            if self.__should_terminate:
                return

            # We should only get here if the state message is non-empty.
            assert(len(state_message) > 0)

            # Print the state message if desired.
            if self.__print_state_messages:
                print("State Message ({}): {}".format(len(state_message), state_message))

            # If we got a valid state message:
            if state_message != b"timeout":
                # Separate the state message into chunks.
                chunks: Mapping = [s.split(":") for s in state_message.decode("UTF-8").split(";")[:-1]]

                # Update the internal state map based on the chunks.
                with self.__state_lock:
                    self.__state_map = dict(chunks)

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

        # While the drone should not terminate:
        while not self.__should_terminate:
            # Attempt to receive a video message from the drone.
            try:
                video_message, source = self.__video_link.socket.recvfrom(2048)
            except socket.timeout:
                video_message, source = b"timeout", self.__video_link.local_endpoint

            # If this unblocks because the drone should terminate, exit.
            if self.__should_terminate:
                return

            # Append the video message to the accumulation buffer.
            accum_buffer += video_message

            # If we reached the end of a frame:
            if len(video_message) != 1460:
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

                # Reset the accumulation buffer so that it is ready to receive more data.
                accum_buffer = b""

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

    def __send_command(self, cmd: str, *, expect_response: bool) -> None:
        """
        Send the specified command to the drone.

        :param cmd:             The command to send.
        :param expect_response: Whether or not we should expect a response from the drone (no for "rc" messages).
        """
        with self.__cmd_lock:
            # Wait for any pending response to a previous command to be received from the drone.
            while self.__response_is_pending:
                self.__no_pending_response.wait(0.1)

                # If this unblocks because the drone should terminate, exit.
                if self.__should_terminate:
                    return

            # Send the command to the drone.
            self.__cmd_link.socket.sendto(bytes(cmd, "utf-8"), self.__cmd_link.remote_endpoint)

            # Print the command that was sent, if desired.
            if self.__print_commands:
                print("Sent Command: {}".format(cmd))

            # If the command requires a response, let the response thread know that it should expect one.
            self.__response_is_pending = expect_response
            if expect_response:
                self.__pending_response.notify()

    # PRIVATE STATIC METHODS

    @staticmethod
    def __rate_to_control_value(rate: float) -> int:
        """
        Convert a floating-point rate to an integer control value in [-100,100].

        :param rate:    The rate.
        :return:        The control value.
        """
        if rate < -1.0:
            rate = -1.0
        if rate > 1.0:
            rate = 1.0
        return int(100 * rate)