class BitsUtil:
    """Utility functions for converting to/from bit strings."""

    # PUBLIC STATIC METHODS

    @staticmethod
    def convert_hilo_bit_string_to_int32(s: str) -> int:
        """
        Convert a binary string, stored with the high bits first, to a 32-bit int.

        .. note::
            The length of the binary string must clearly be <= 32.
        .. note::
            As an example, "00000000000000000000000000001100" gets converted to 12.

        :param s:   The binary string.
        :return:    The 32-bit int.
        """
        if len(s) <= 32:
            return int(s, 2)
        else:
            raise ValueError("Cannot convert a binary string whose length is > 32 to a 32-bit int")

    @staticmethod
    def convert_int32_to_hilo_bit_string(i: int) -> str:
        """
        Convert a 32-bit int to a 32-bit binary string, with the high bits first.

        .. note::
            As an example, 12 gets converted to "00000000000000000000000000001100".

        :param i:   The 32-bit int.
        :return:    The 32-bit string.
        """
        return f"{i:032b}"

    @staticmethod
    def convert_int32_to_lohi_bit_string(i: int) -> str:
        """
        Convert a 32-bit int to a 32-bit binary string, with the low bits first.

        .. note::
            As an example, 12 gets converted to "00110000000000000000000000000000".

        :param i:   The 32-bit int.
        :return:    The 32-bit string.
        """
        return BitsUtil.convert_int32_to_hilo_bit_string(i)[::-1]

    @staticmethod
    def convert_lohi_bit_string_to_int32(s: str) -> int:
        """
        Convert a binary string, stored with the low bits first, to a 32-bit int.

        .. note::
            The length of the binary string must clearly be <= 32.
        .. note::
            As an example, "00110000000000000000000000000000" gets converted to 12.

        :param s:   The binary string.
        :return:    The 32-bit int.
        """
        return BitsUtil.convert_hilo_bit_string_to_int32(s[::-1])
