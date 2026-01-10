#!/usr/bin/env python3
"""
Script to extract the robot model library from a real Franka Panda robot.

This connects to a real robot and downloads the model shared object file
that contains the kinematics and dynamics functions (Jacobians, mass matrix,
Coriolis, gravity, etc.). The model file can then be used by the simulator.

Usage:
    python -m franka_sim.extract_model <robot_ip> [--output-dir <dir>]

Example:
    python -m franka_sim.extract_model 192.168.1.100
"""

import argparse
import platform
import socket
import struct
from pathlib import Path

# Protocol constants from libfranka 0.9.2
COMMAND_PORT = 1337
PROTOCOL_VERSION = 5


# Command enum values (from service_types.h)
COMMAND_CONNECT = 0
COMMAND_LOAD_MODEL_LIBRARY = 13

# Architecture enum (from service_types.h)
ARCH_X64 = 0
ARCH_X86 = 1
ARCH_ARM = 2
ARCH_ARM64 = 3

# System enum (from service_types.h)
SYSTEM_LINUX = 0
SYSTEM_WINDOWS = 1


def get_architecture():
    """Determine current system architecture"""
    machine = platform.machine().lower()
    if machine in ("x86_64", "amd64"):
        return ARCH_X64
    elif machine in ("i386", "i686", "x86"):
        return ARCH_X86
    elif machine in ("aarch64", "arm64"):
        return ARCH_ARM64
    elif machine.startswith("arm"):
        return ARCH_ARM
    else:
        raise RuntimeError(f"Unsupported architecture: {machine}")


def get_system():
    """Determine current operating system"""
    system = platform.system().lower()
    if system == "linux":
        return SYSTEM_LINUX
    elif system == "windows":
        return SYSTEM_WINDOWS
    else:
        raise RuntimeError(f"Unsupported operating system: {system}")


def receive_exact(sock: socket.socket, size: int) -> bytes:
    """Receive exactly `size` bytes from socket"""
    data = bytearray()
    while len(data) < size:
        chunk = sock.recv(size - len(data))
        if not chunk:
            raise ConnectionError("Connection closed while receiving data")
        data.extend(chunk)
    return bytes(data)


def extract_model(robot_ip: str, output_dir: Path) -> Path:
    """
    Connect to robot and extract the model library.

    Args:
        robot_ip: IP address of the Franka robot
        output_dir: Directory to save the model file

    Returns:
        Path to the saved model file
    """
    arch = get_architecture()
    system = get_system()

    arch_names = {ARCH_X64: "x64", ARCH_X86: "x86", ARCH_ARM: "arm", ARCH_ARM64: "arm64"}
    system_names = {SYSTEM_LINUX: "linux", SYSTEM_WINDOWS: "windows"}
    suffix = ".so" if system == SYSTEM_LINUX else ".dll"

    print(f"Architecture: {arch_names[arch]}")
    print(f"System: {system_names[system]}")
    print(f"Connecting to robot at {robot_ip}:{COMMAND_PORT}...")

    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.settimeout(10.0)

    try:
        sock.connect((robot_ip, COMMAND_PORT))
        print("Connected!")

        # Step 1: Send Connect command
        print("Sending Connect command...")
        command_id = 1

        # Connect request: version (uint16) + udp_port (uint16)
        # We don't need UDP for this, so use port 0
        connect_payload = struct.pack("<HH", PROTOCOL_VERSION, 0)
        connect_header = struct.pack("<III", COMMAND_CONNECT, command_id, 12 + len(connect_payload))

        sock.sendall(connect_header + connect_payload)

        # Receive Connect response
        response_header = receive_exact(sock, 12)
        resp_command, resp_id, resp_size = struct.unpack("<III", response_header)

        if resp_command != COMMAND_CONNECT or resp_id != command_id:
            raise RuntimeError(f"Unexpected response: command={resp_command}, id={resp_id}")

        payload_size = resp_size - 12
        response_payload = receive_exact(sock, payload_size) if payload_size > 0 else b""

        # Parse response based on actual size
        # Response format from 0.9.2: status (uint8) + version (uint16) = 3 bytes minimum
        # But robot might send with padding
        if len(response_payload) >= 4:
            # 4+ bytes: try status (uint8) + padding (1 byte) + version (uint16)
            # or status (uint16) + version (uint16)
            status = response_payload[0]
            version = struct.unpack("<H", response_payload[2:4])[0]
        elif len(response_payload) >= 3:
            # 3 bytes: status (uint8) + version (uint16)
            status = response_payload[0]
            version = struct.unpack("<H", response_payload[1:3])[0]
        elif len(response_payload) >= 1:
            # Just status
            status = response_payload[0]
            version = 0
        else:
            raise RuntimeError(f"Response payload too small: {len(response_payload)} bytes")

        if status != 0:
            if status == 1:
                raise RuntimeError(
                    f"Incompatible library version. Robot version: {version}, "
                    f"our version: {PROTOCOL_VERSION}"
                )
            raise RuntimeError(f"Connect failed with status: {status}")

        print(f"Connected! Robot protocol version: {version}")

        # Step 2: Send LoadModelLibrary command
        print("Requesting model library...")
        command_id = 2

        # LoadModelLibrary request: architecture (uint8) + system (uint8)
        model_payload = struct.pack("<BB", arch, system)
        model_header = struct.pack(
            "<III", COMMAND_LOAD_MODEL_LIBRARY, command_id, 12 + len(model_payload)
        )

        sock.sendall(model_header + model_payload)

        # Receive LoadModelLibrary response header
        response_header = receive_exact(sock, 12)
        resp_command, resp_id, resp_size = struct.unpack("<III", response_header)

        if resp_command != COMMAND_LOAD_MODEL_LIBRARY or resp_id != command_id:
            raise RuntimeError(f"Unexpected response: command={resp_command}, id={resp_id}")

        # Receive full response (status + model data)
        payload_size = resp_size - 12
        print(f"Receiving model data ({payload_size} bytes)...")
        response_payload = receive_exact(sock, payload_size)

        # First byte is status (no padding in packed struct)
        status = response_payload[0]

        if status != 0:
            raise RuntimeError(f"LoadModelLibrary failed with status: {status}")

        # Rest is the model data (after 1 byte status - no padding!)
        model_data = response_payload[1:]
        print(f"Received model library: {len(model_data)} bytes")

        # Save the model file
        output_dir.mkdir(parents=True, exist_ok=True)
        model_filename = f"libfcimodels_{arch_names[arch]}{suffix}"
        model_path = output_dir / model_filename

        with open(model_path, "wb") as f:
            f.write(model_data)

        print(f"Saved model to: {model_path}")
        return model_path

    finally:
        sock.close()


def main():
    parser = argparse.ArgumentParser(
        description="Extract robot model library from a real Franka Panda robot"
    )
    parser.add_argument("robot_ip", help="IP address of the Franka robot")
    parser.add_argument(
        "--output-dir",
        "-o",
        type=Path,
        default=Path(__file__).parent / "models",
        help="Directory to save the model file (default: franka_sim/models)",
    )
    args = parser.parse_args()

    try:
        model_path = extract_model(args.robot_ip, args.output_dir)
        print(f"\nSuccess! Model library saved to: {model_path}")
        print("\nYou can now use this model with the Franka simulator.")
    except Exception as e:
        print(f"\nError: {e}")
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
