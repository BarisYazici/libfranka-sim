"""Readiness probe for a running franka-sim server, for CI and containers.

Not a port scan: the probe performs the real libfranka v10 Connect handshake
over TCP and then waits for one RobotState datagram on UDP, so a passing check
means the server is actually serving the FCI — accepting clients, agreeing on
the protocol version and streaming state — not merely holding the port open.

Used as the Docker image's readiness gate, as the GitHub Action's readiness
wait, and directly in user CI via the ``franka-sim-check`` console script::

    franka-sim-check --host 127.0.0.1 --port 1337 --timeout 30

Exit code 0 means healthy; 1 means the server could not be reached, refused
or garbled the handshake, or never streamed a state within the timeout.

Note: the probe occupies the server's single FCI client slot while it runs
(a fraction of a second), exactly like a real ``franka::Robot`` connection.
"""

import argparse
import socket
import struct
import sys
import time
from dataclasses import dataclass
from typing import Optional, Sequence

from franka_sim.franka_protocol import COMMAND_PORT, Command, ConnectStatus, MessageHeader

#: research_interface::robot kVersion this probe announces; must match
#: FrankaSimServer.library_version.
PROTOCOL_VERSION = 10

#: 12-byte MessageHeader + struct.pack("<HH", version, udp_port).
_CONNECT_REQUEST_SIZE = 12 + 4

#: Header (12) + status uint8 + version uint16: the least a well-formed
#: Connect response can be.
_MIN_CONNECT_RESPONSE_SIZE = 12 + 3


class HealthCheckError(Exception):
    """The server failed the probe; the message says at which stage."""


@dataclass
class HealthReport:
    """What a successful probe observed."""

    server_version: int
    state_bytes: int


def _recv_exactly(sock: socket.socket, count: int, deadline: float, what: str) -> bytes:
    """Read exactly ``count`` bytes before ``deadline`` or raise HealthCheckError."""
    chunks = []
    remaining = count
    while remaining > 0:
        sock.settimeout(_remaining_or_raise(deadline, f"reading {what}"))
        try:
            chunk = sock.recv(remaining)
        except socket.timeout as exc:
            raise HealthCheckError(f"timed out reading {what}") from exc
        except OSError as exc:
            raise HealthCheckError(f"connection error reading {what}: {exc}") from exc
        if not chunk:
            raise HealthCheckError(f"server closed the connection while sending {what}")
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


def check_server(
    host: str = "127.0.0.1",
    port: int = COMMAND_PORT,
    timeout: float = 30.0,
) -> HealthReport:
    """Probe one franka-sim server; return a HealthReport or raise HealthCheckError.

    ``timeout`` bounds the whole probe: TCP connect retries (the server may
    still be starting up), the Connect exchange, and the wait for the first
    RobotState datagram. Every failure mode — unreachable, refused, malformed
    response, silent UDP — surfaces as HealthCheckError; no other exception
    escapes short of a bug.
    """
    deadline = time.monotonic() + timeout

    tcp = _connect_with_retries(host, port, deadline)
    try:
        udp = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        try:
            udp.bind(("", 0))
            udp_port = udp.getsockname()[1]

            tcp.settimeout(_remaining_or_raise(deadline, "sending the Connect request"))
            payload = struct.pack("<HH", PROTOCOL_VERSION, udp_port)
            header = MessageHeader(
                command=Command.kConnect, command_id=1, size=_CONNECT_REQUEST_SIZE
            )
            try:
                tcp.sendall(header.to_bytes() + payload)
            except OSError as exc:
                raise HealthCheckError(f"failed to send Connect request: {exc}") from exc

            raw_header = _recv_exactly(tcp, 12, deadline, "Connect response header")
            try:
                response_header = MessageHeader.from_bytes(raw_header)
            except (ValueError, struct.error) as exc:
                raise HealthCheckError(
                    f"malformed Connect response header (is {host}:{port} really a "
                    f"franka-sim / FCI port?): {exc}"
                ) from exc
            if response_header.size < _MIN_CONNECT_RESPONSE_SIZE:
                raise HealthCheckError(
                    f"Connect response claims {response_header.size} bytes, less than "
                    f"the minimum well-formed response ({_MIN_CONNECT_RESPONSE_SIZE})"
                )
            body = _recv_exactly(tcp, response_header.size - 12, deadline, "Connect response body")
            status, server_version = struct.unpack("<BH", body[:3])
            if status != ConnectStatus.kSuccess:
                raise HealthCheckError(
                    f"server refused the Connect handshake "
                    f"(status={status}, server version={server_version})"
                )

            state_bytes = _await_robot_state(udp, tcp, deadline, udp_port)
            return HealthReport(server_version=server_version, state_bytes=state_bytes)
        finally:
            udp.close()
    finally:
        # Close abruptly rather than half-shutdown: the server treats a closed
        # command socket as the client leaving, which is what we are.
        tcp.close()


def _await_robot_state(
    udp: socket.socket, tcp: socket.socket, deadline: float, udp_port: int
) -> int:
    """Wait for a RobotState datagram from the server we handshook with.

    Datagrams from any other sender (a stray packet on an unprivileged UDP
    port must not green-light the probe) are discarded and the wait continues.
    """
    server_ip = tcp.getpeername()[0]
    while True:
        udp.settimeout(
            _remaining_or_raise(deadline, f"waiting for a RobotState datagram on UDP {udp_port}")
        )
        try:
            datagram, sender = udp.recvfrom(65535)
        except socket.timeout as exc:
            raise HealthCheckError(
                "handshake succeeded but no RobotState datagram arrived on UDP "
                f"port {udp_port} before the timeout"
            ) from exc
        if sender[0] == server_ip and datagram:
            return len(datagram)


def _remaining_or_raise(deadline: float, doing: str) -> float:
    remaining = deadline - time.monotonic()
    if remaining <= 0:
        raise HealthCheckError(f"timed out {doing}")
    return remaining


def _connect_with_retries(host: str, port: int, deadline: float) -> socket.socket:
    """TCP-connect, retrying until ``deadline`` — the server may still be booting."""
    last_error: Optional[Exception] = None
    while True:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise HealthCheckError(
                f"could not connect to {host}:{port} before the timeout"
                + (f" (last error: {last_error})" if last_error else "")
            )
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(min(2.0, remaining))
        try:
            sock.connect((host, port))
            return sock
        except OSError as exc:
            last_error = exc
            sock.close()
            time.sleep(min(0.25, max(0.0, deadline - time.monotonic())))


def main(argv: Optional[Sequence[str]] = None) -> int:
    """CLI entry point (``franka-sim-check``). Returns the process exit code."""
    parser = argparse.ArgumentParser(
        description="Probe a franka-sim server: real Connect handshake + one "
        "RobotState datagram. Exit 0 iff the server is serving the FCI."
    )
    parser.add_argument("--host", default="127.0.0.1", help="Server address (default: 127.0.0.1)")
    parser.add_argument(
        "--port", type=int, default=COMMAND_PORT, help=f"FCI command port (default: {COMMAND_PORT})"
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=30.0,
        help="Overall budget in seconds, including connect retries while the "
        "server boots (default: 30)",
    )
    args = parser.parse_args(argv)

    try:
        report = check_server(args.host, args.port, timeout=args.timeout)
    except HealthCheckError as exc:
        print(f"franka-sim-check: {exc}", file=sys.stderr)
        return 1
    print(
        f"ok: server at {args.host}:{args.port} speaks protocol v{report.server_version} "
        f"and streamed a {report.state_bytes}-byte RobotState"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
