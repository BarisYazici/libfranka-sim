import logging
import socket
import threading
import time
from unittest.mock import Mock

import numpy as np
import pytest

from franka_sim.franka_protocol import COMMAND_PORT

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Mock robot state for testing
MOCK_ROBOT_STATE = {"q": np.zeros(7), "dq": np.zeros(7), "tau_J": np.zeros(7)}


def wait_for_server(port, max_retries=20, retry_delay=0.2):
    """Helper function to wait for server to start"""
    for attempt in range(max_retries):
        try:
            logger.info(f"Attempt {attempt + 1}/{max_retries} to connect to server...")
            test_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            test_socket.settimeout(1.0)
            test_socket.connect(("localhost", port))
            test_socket.close()
            logger.info("Successfully connected to server")
            return True
        except (ConnectionRefusedError, socket.timeout) as e:
            logger.warning(f"Connection attempt failed: {e}")
            if attempt < max_retries - 1:
                time.sleep(retry_delay)
            continue
        except Exception as e:
            logger.error(f"Unexpected error while connecting: {e}")
            if attempt < max_retries - 1:
                time.sleep(retry_delay)
            continue
    return False


@pytest.fixture
def mock_genesis_sim():
    """Fixture that provides a mocked Genesis simulator"""
    mock_sim = Mock()
    mock_sim.get_robot_state.return_value = MOCK_ROBOT_STATE
    return mock_sim


@pytest.fixture
def sim_server(mock_genesis_sim):
    """Fixture that provides a server with mocked Genesis simulator"""
    # First ensure no existing server is running
    try:
        test_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        test_socket.settimeout(1.0)
        test_socket.connect(("localhost", COMMAND_PORT))
        test_socket.close()
        raise RuntimeError("Port already in use. Make sure no other server is running.")
    except (ConnectionRefusedError, socket.timeout):
        pass

    from franka_sim.franka_sim_server import FrankaSimServer

    server = FrankaSimServer(enable_vis=False, genesis_sim=mock_genesis_sim)
    server_thread = threading.Thread(target=server.run_server)
    server_thread.daemon = True

    try:
        logger.info("Starting server thread...")
        server_thread.start()
        logger.info("Waiting for server to start...")

        if not wait_for_server(COMMAND_PORT):
            logger.error("Server failed to start after maximum retries")
            server.stop()
            server_thread.join(timeout=1.0)
            raise RuntimeError("Server failed to start")

        logger.info("Server started successfully")
        yield server

    finally:
        logger.info("Cleaning up server...")
        server.stop()
        server_thread.join(timeout=2.0)  # Increased timeout for cleanup
        # Ensure all sockets are closed
        if hasattr(server, "client_socket") and server.client_socket:
            try:
                server.client_socket.shutdown(socket.SHUT_RDWR)
            except socket.error:
                pass
            try:
                server.client_socket.close()
            except socket.error:
                pass
        if hasattr(server, "udp_socket") and server.udp_socket:
            try:
                server.udp_socket.close()
            except socket.error:
                pass
        if hasattr(server, "server_socket") and server.server_socket:
            try:
                server.server_socket.shutdown(socket.SHUT_RDWR)
            except socket.error:
                pass
            try:
                server.server_socket.close()
            except socket.error:
                pass
        # Wait for sockets to fully close
        time.sleep(0.2)


@pytest.fixture
def tcp_client():
    """Fixture that provides a TCP client socket"""
    client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    client.settimeout(5.0)  # Increased timeout for reliability
    yield client
    try:
        client.shutdown(socket.SHUT_RDWR)
    except socket.error:
        pass
    try:
        client.close()
    except socket.error:
        pass


@pytest.fixture
def udp_client():
    """Fixture that provides a UDP client socket"""
    client = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    client.settimeout(5.0)  # Increased timeout for reliability
    yield client
    try:
        client.close()
    except socket.error:
        pass


@pytest.fixture(autouse=True)
def cleanup_sockets():
    """Fixture to ensure sockets are cleaned up after each test"""
    yield
    # Clean up any lingering sockets in TIME_WAIT state
    time.sleep(0.2)  # Increased delay to allow socket cleanup
