import asyncio
from contextlib import AbstractAsyncContextManager
from socket import socket

from stpr.channels import Channel


class Connection:
    """
    Represents a socket connection.

    A connection is a bidirectional entity that can be used to send and receive data over
    a socket. Connection instances are created through the :func:`Socket` class.
    """
    def __init__(self, reader, writer) -> None:
        """
        Constructs a connection object.

        :param reader: The connection reader.
        :param writer: The connection writer.
        """
        self.reader = reader
        self.writer = writer

    def get_peer_addr(self) -> str:
        """
        Returns the remote address to which the socket is connected to.

        :return: A string with the remote address.
        """
        return self.writer.transport.get_extra_info('peername')

    async def readline(self):
        """
        Reads one line from the reader.

        :return: The contents of the line.
        """
        line_bytes = await self.reader.readline()
        return line_bytes.decode('utf-8')

    async def write_chunk(self, chunk) -> None:
        """
        Writes a chunk of data to this connection.

        :param chunk: The chunk to write.
        """
        self.writer.write(chunk.encode('utf-8'))
        await self.writer.drain()

    async def close(self) -> None:
        """
        Closes this connection.
        """
        self.reader.close()
        self.writer.close()


class ServerSocket(AbstractAsyncContextManager):
    """
    Wraps an asynchronous server socket.

    This is a convenience class that allows a server socket to be iterated over the connections:

    .. code-block:: Python

        with ServerSocket('0.0.0.0', 8000, family=socket.AF_) as s:
            with stpr.parallelFor(s.connections()) as c:
                handle_connection(c)

    """

    def __init__(self, address: str, port: int, family=socket.AF_INET):
        """
        Instantiates a `ServerSocket`.

        :param address: The address to bind the socket to.
        :param port: The port to start the socket on.
        :param family: An optional socket family. The default is :const:`socket.AF_INET`.
        """
        self.address = address
        self.port = port
        self.family = family
        self._connections = Channel()

    async def __aenter__(self):
        self.server = await asyncio.start_server(self._handle_connection, host=self.address,
                                                 port=self.port, family=self.family)
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        self._connections.close()
        self.server.close()

    def _handle_connection(self, reader, writer):
        self._connections._append_now(Connection(reader, writer))

    def connections(self) -> Channel[Connection]:
        """
        Returns a channel with the incoming connections.

        :return: A :class:`stpr.Channel` containing all connections initiated on this socket. The
            channel is closed when the socket is closed.
        """
        return self._connections
