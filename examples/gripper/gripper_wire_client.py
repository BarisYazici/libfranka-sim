"""Minimal libfranka-compatible gripper wire client (research_interface::gripper v3).

Speaks the same TCP/UDP protocol as franka::Gripper: TCP port 1338 for
Connect/Homing/Move/Grasp/Stop (one response each) and a one-way UDP
GripperState broadcast. Used by the physics integration test and the viewer demo
to drive the simulated hand exactly as a real client would.
"""

import socket
import struct
import threading

from franka_sim.gripper_protocol import (
    GRIPPER_COMMAND_PORT,
    GRIPPER_HEADER_SIZE,
    GRIPPER_STATE_SIZE,
    GRIPPER_VERSION,
    GripperCommand,
    GripperCommandHeader,
    GripperConnectStatus,
    GripperState,
    GripperStatus,
)


class GripperWireClient:
    def __init__(self, host="127.0.0.1", port=GRIPPER_COMMAND_PORT):
        self.host = host
        self.port = port
        self.tcp = None
        self.udp = None
        self._cmd_id = 0
        self._latest_state = None
        self._state_lock = threading.Lock()
        self._rx_thread = None
        self._running = False

    def _next_id(self):
        self._cmd_id += 1
        return self._cmd_id

    def _recv_exact(self, n):
        buf = b""
        while len(buf) < n:
            chunk = self.tcp.recv(n - len(buf))
            if not chunk:
                raise ConnectionError("gripper server closed the connection")
            buf += chunk
        return buf

    def _send(self, command, payload=b""):
        header = GripperCommandHeader(command, self._next_id(), GRIPPER_HEADER_SIZE + len(payload))
        self.tcp.sendall(header.to_bytes() + payload)
        rh = GripperCommandHeader.from_bytes(self._recv_exact(GRIPPER_HEADER_SIZE))
        body = self._recv_exact(rh.size - GRIPPER_HEADER_SIZE)
        return rh, body

    def connect(self):
        # Bind a UDP socket FIRST so we can tell the server which port to broadcast to.
        self.udp = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.udp.bind(("0.0.0.0", 0))
        udp_port = self.udp.getsockname()[1]
        self.tcp = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.tcp.connect((self.host, self.port))
        self.tcp.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        _, body = self._send(GripperCommand.kConnect, struct.pack("<HH", GRIPPER_VERSION, udp_port))
        status, _version = struct.unpack("<HH", body[:4])
        if status != GripperConnectStatus.kSuccess:
            raise RuntimeError(f"gripper Connect failed: status={status}")
        self._running = True
        self._rx_thread = threading.Thread(target=self._rx_state, daemon=True)
        self._rx_thread.start()

    def _rx_state(self):
        self.udp.settimeout(0.5)
        while self._running:
            try:
                data, _ = self.udp.recvfrom(GRIPPER_STATE_SIZE)
            except socket.timeout:
                continue
            except OSError:
                break
            if len(data) >= GRIPPER_STATE_SIZE:
                _id, st = GripperState.unpack(data)
                with self._state_lock:
                    self._latest_state = st

    def read_state(self):
        with self._state_lock:
            return self._latest_state

    @staticmethod
    def _status(body):
        return GripperStatus(struct.unpack("<H", body[:2])[0])

    def homing(self):
        return self._status(self._send(GripperCommand.kHoming)[1])

    def move(self, width, speed):
        return self._status(self._send(GripperCommand.kMove, struct.pack("<dd", width, speed))[1])

    def grasp(self, width, epsilon_inner, epsilon_outer, speed, force):
        payload = struct.pack("<ddddd", width, epsilon_inner, epsilon_outer, speed, force)
        return self._status(self._send(GripperCommand.kGrasp, payload)[1])

    def stop(self):
        return self._status(self._send(GripperCommand.kStop)[1])

    def close(self):
        self._running = False
        if self._rx_thread:
            self._rx_thread.join(timeout=1.0)
        for s in (self.tcp, self.udp):
            try:
                if s:
                    s.close()
            except OSError:
                pass
