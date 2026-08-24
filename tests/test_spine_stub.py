import http.client
import json
import shutil
import threading

import pytest

from franka_sim.mobile.spine_stub import SpineError, SpineModel, SpineStubServer


class FakeClock:
    """Manually advanced monotonic clock."""

    def __init__(self):
        self.now = 0.0

    def __call__(self):
        return self.now

    def advance(self, seconds):
        self.now += seconds


@pytest.fixture
def clock():
    return FakeClock()


@pytest.fixture
def model(clock):
    return SpineModel(clock=clock)


@pytest.fixture
def stub():
    """A plain-HTTP stub on an ephemeral port."""
    server = SpineStubServer(host="127.0.0.1", port=0)
    server.start()
    yield server
    server.stop()


def request(server, method, endpoint, body=None):
    """Issue one request the way SpineApiClient does and return (status, payload)."""
    connection = http.client.HTTPConnection("127.0.0.1", server.port, timeout=5)
    payload = json.dumps(body) if body is not None else None
    connection.request(
        method, f"/spine/api/{endpoint}", body=payload, headers={"Content-Type": "application/json"}
    )
    response = connection.getresponse()
    raw = response.read()
    connection.close()
    return response.status, (json.loads(raw) if raw else None)


# --- motion model ----------------------------------------------------------


def test_model_starts_switched_off_at_zero(model):
    assert model.state == "SwitchedOff"
    assert model.position_m() == pytest.approx(0.0)


def test_switch_on_enables_motion(model):
    assert model.switch_on() == "SwitchedOn"
    model.start_motion(0.4, 0.05)


def test_motion_is_rejected_while_switched_off(model):
    with pytest.raises(SpineError) as error:
        model.start_motion(0.4, 0.05)
    assert error.value.status == 409


@pytest.mark.parametrize("target", [-0.01, 0.86, 1.5])
def test_motion_outside_the_limits_is_rejected(model, target):
    model.switch_on()
    with pytest.raises(SpineError) as error:
        model.start_motion(target, 0.05)
    assert error.value.status == 400


def test_non_positive_velocity_is_rejected(model):
    model.switch_on()
    with pytest.raises(SpineError) as error:
        model.start_motion(0.4, 0.0)
    assert error.value.status == 400


def test_position_advances_at_constant_velocity(model, clock):
    model.switch_on()
    model.start_motion(0.5, 0.1)
    clock.advance(1.0)
    assert model.position_m() == pytest.approx(0.1)
    clock.advance(1.0)
    assert model.position_m() == pytest.approx(0.2)


def test_position_stops_at_the_target(model, clock):
    model.switch_on()
    model.start_motion(0.2, 0.1)
    clock.advance(60.0)
    assert model.position_m() == pytest.approx(0.2)
    assert model.is_moving() is False


def test_motion_downwards_decreases_the_position(model, clock):
    model.switch_on()
    model.start_motion(0.6, 1.0)
    clock.advance(1.0)
    model.start_motion(0.1, 0.1)
    clock.advance(1.0)
    assert model.position_m() == pytest.approx(0.5)


def test_halt_freezes_at_the_current_position(model, clock):
    model.switch_on()
    model.start_motion(0.8, 0.1)
    clock.advance(2.0)
    model.halt()
    clock.advance(10.0)
    assert model.position_m() == pytest.approx(0.2)
    assert model.is_moving() is False


def test_fault_reset_clears_a_fault(model):
    model.trigger_fault()
    assert model.state == "Fault"
    assert model.fault_reset() == "SwitchedOff"


def test_switch_on_is_rejected_while_faulted(model):
    model.trigger_fault()
    with pytest.raises(SpineError) as error:
        model.switch_on()
    assert error.value.status == 409


def test_parameters_report_the_urdf_limits(model):
    assert model.parameters() == {"user_limits": {"lower_limit_in_mm": 0, "upper_limit_in_mm": 850}}


# --- HTTP surface ------------------------------------------------------------


def test_state_is_a_bare_json_string(stub):
    status, payload = request(stub, "GET", "state")
    assert status == 200
    assert payload == "SwitchedOff"
    assert isinstance(payload, str)


def test_position_is_reported_in_integer_millimetres(stub):
    request(stub, "POST", "spine:switch-on")
    request(stub, "POST", "motion-mm:start", {"position": 400, "velocity": 50})
    status, payload = request(stub, "GET", "position-mm")
    assert status == 200
    assert isinstance(payload["position"], int)


def test_switch_on_off_round_trip(stub):
    assert request(stub, "POST", "spine:switch-on") == (200, "SwitchedOn")
    assert request(stub, "GET", "state") == (200, "SwitchedOn")
    assert request(stub, "POST", "spine:switch-off") == (200, "SwitchedOff")
    assert request(stub, "GET", "state") == (200, "SwitchedOff")


def test_motion_start_returns_stop_by(stub):
    request(stub, "POST", "spine:switch-on")
    status, payload = request(
        stub,
        "POST",
        "motion-mm:start",
        {"position": 400, "velocity": 50, "acceleration": 100, "deceleration": 100},
    )
    assert status == 200
    assert payload["StopBy"] == "TargetReached"


def test_motion_start_is_rejected_while_switched_off(stub):
    status, _ = request(stub, "POST", "motion-mm:start", {"position": 400, "velocity": 50})
    assert status == 409


def test_motion_start_beyond_the_upper_limit_is_rejected(stub):
    request(stub, "POST", "spine:switch-on")
    status, _ = request(stub, "POST", "motion-mm:start", {"position": 900, "velocity": 50})
    assert status == 400


def test_halt_returns_the_current_state(stub):
    request(stub, "POST", "spine:switch-on")
    assert request(stub, "POST", "motion:halt") == (200, "SwitchedOn")


def test_fault_reset_endpoint_returns_a_state(stub):
    status, payload = request(stub, "POST", "spine:fault-reset")
    assert status == 200
    assert isinstance(payload, str)


def test_parameters_endpoint_uses_millimetre_keys(stub):
    status, payload = request(stub, "GET", "parameters")
    assert status == 200
    assert payload["user_limits"]["lower_limit_in_mm"] == 0
    assert payload["user_limits"]["upper_limit_in_mm"] == 850


def test_unknown_endpoint_returns_404(stub):
    status, _ = request(stub, "GET", "nonexistent")
    assert status == 404


def test_position_moves_over_wall_clock_time(stub):
    """The default clock is real time: a slow move is observable from outside."""
    request(stub, "POST", "spine:switch-on")
    request(stub, "POST", "motion-mm:start", {"position": 800, "velocity": 100})
    _, first = request(stub, "GET", "position-mm")
    threading.Event().wait(0.3)
    _, second = request(stub, "GET", "position-mm")
    assert second["position"] > first["position"]


# --- upstream client compatibility -------------------------------------------


def test_the_real_spine_api_client_shape_is_satisfied(stub):
    """Replicates SpineApiClient's unit conversions against the stub."""
    requests = pytest.importorskip("requests")

    session = requests.Session()
    session.headers.update({"Content-Type": "application/json"})
    session.verify = False
    base_url = f"http://127.0.0.1:{stub.port}/spine/api"

    assert session.post(f"{base_url}/spine:switch-on", json=None, timeout=5).json() == "SwitchedOn"

    started = session.post(
        f"{base_url}/motion-mm:start",
        json={
            "position": int(round(0.4 * 1000)),
            "velocity": int(round(0.05 * 1000)),
            "acceleration": int(round(0.1 * 1000)),
            "deceleration": int(round(0.1 * 1000)),
        },
        timeout=5,
    ).json()
    assert started["StopBy"] == "TargetReached"

    position = session.get(f"{base_url}/position-mm", timeout=5).json()
    position["position"] = position["position"] / 1000.0
    assert 0.0 <= position["position"] <= 0.85

    parameters = session.get(f"{base_url}/parameters", timeout=5).json()
    limits = parameters["user_limits"]
    limits["lower_limit"] = float(limits.pop("lower_limit_in_mm", 0)) / 1000.0
    limits["upper_limit"] = float(limits.pop("upper_limit_in_mm", 0)) / 1000.0
    assert limits["lower_limit"] == pytest.approx(0.0)
    assert limits["upper_limit"] == pytest.approx(0.85)

    assert session.get(f"{base_url}/state", timeout=5).json() == "SwitchedOn"


@pytest.mark.skipif(shutil.which("openssl") is None, reason="openssl CLI is required")
def test_https_mode_serves_a_self_signed_certificate(tmp_path):
    """franka_spine_server talks HTTPS with verify=False; prove the TLS path works."""
    import ssl

    from franka_sim.mobile.spine_stub import make_self_signed_cert

    certfile, keyfile = make_self_signed_cert(tmp_path)
    server = SpineStubServer(host="127.0.0.1", port=0, certfile=certfile, keyfile=keyfile)
    server.start()
    try:
        context = ssl.create_default_context()
        context.check_hostname = False
        context.verify_mode = ssl.CERT_NONE
        connection = http.client.HTTPSConnection(
            "127.0.0.1", server.port, context=context, timeout=5
        )
        connection.request("GET", "/spine/api/state")
        response = connection.getresponse()
        payload = json.loads(response.read())
        connection.close()
        assert response.status == 200
        assert payload == "SwitchedOff"
    finally:
        server.stop()
