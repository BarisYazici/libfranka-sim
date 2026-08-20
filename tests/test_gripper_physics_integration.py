import sys
import threading
import time

import pytest

# --- Genesis availability check -----------------------------------------------
# conftest.py stubs `genesis` with a MagicMock so that protocol/handshake tests
# can import franka_sim_server without Genesis installed. Real-Genesis tests
# must swap out both the top-level stub and any genesis.* submodules; otherwise
# two real Genesis module instances can coexist and break gs.init().


def _pop_genesis_modules():
    saved = {}
    for name in list(sys.modules):
        if name == "genesis" or name.startswith("genesis."):
            saved[name] = sys.modules.pop(name)
    return saved


def _restore_genesis_modules(saved):
    for name in list(sys.modules):
        if name == "genesis" or name.startswith("genesis."):
            sys.modules.pop(name)
    sys.modules.update(saved)


def _genesis_available():
    import importlib.util

    saved = _pop_genesis_modules()
    try:
        spec = importlib.util.find_spec("genesis")
        return spec is not None and isinstance(getattr(spec, "origin", None), str)
    finally:
        _restore_genesis_modules(saved)


genesis_required = pytest.mark.skipif(
    not _genesis_available(), reason="Genesis is not installed; skipping physics integration tests."
)


@pytest.fixture
def real_genesis():
    saved = _pop_genesis_modules()
    fsm = None
    original_gs = None
    try:
        try:
            import genesis as gs
        except ImportError:
            pytest.skip("Genesis is not installed; skipping physics integration tests.")

        if not isinstance(getattr(gs, "__file__", None), str):
            pytest.skip("real genesis could not be loaded")

        import franka_sim.franka_genesis_sim as fsm

        original_gs = fsm.gs
        fsm.gs = gs
        yield gs
    finally:
        if fsm is not None:
            fsm.gs = original_gs
        _restore_genesis_modules(saved)


def _wait_until(pred, timeout=10.0, dt=0.05):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if pred():
            return True
        time.sleep(dt)
    return False


@pytest.fixture
def physics_gripper(real_genesis):
    """A running 9-DOF sim + gripper server with the physics backend."""
    from franka_sim.franka_genesis_sim import FrankaGenesisSim
    from franka_sim.gripper_physics import GenesisFrankaHand
    from franka_sim.gripper_server import FrankaGripperServer

    sim = FrankaGenesisSim(enable_vis=False, enable_hand=True)
    sim.initialize_simulation()
    sim.running = True
    sim_thread = threading.Thread(target=sim.run_simulation, daemon=True)
    sim_thread.start()

    server = FrankaGripperServer(host="127.0.0.1", port=0, backend=GenesisFrankaHand(sim))
    srv_thread = threading.Thread(target=server.run_server, daemon=True)
    srv_thread.start()
    assert _wait_until(lambda: server.running and server.server_socket is not None)
    port = server.server_socket.getsockname()[1]
    yield port
    server.stop()
    sim.stop()
    srv_thread.join(timeout=2.0)
    sim_thread.join(timeout=2.0)


@genesis_required
def test_wire_client_move_drives_physics_fingers(physics_gripper):
    """A libfranka-style wire client opens/closes the physics fingers and sees it over UDP."""
    from examples.gripper.gripper_wire_client import GripperWireClient

    client = GripperWireClient(port=physics_gripper)
    client.connect()
    try:
        client.homing()
        # move() blocks server-side until the fingers settle; then UDP state is fresh.
        assert client.move(0.08, 0.1).name == "kSuccess"
        assert _wait_until(lambda: client.read_state() and client.read_state().width > 0.06)
        assert client.move(0.0, 0.1).name == "kSuccess"
        assert _wait_until(lambda: client.read_state() and client.read_state().width < 0.02)
    finally:
        client.close()
