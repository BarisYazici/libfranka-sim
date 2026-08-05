"""A fake Franka Spine REST device, good enough for franka_spine_server.

Upstream's ``franka_spine_server`` talks to an HTTPS REST API on the mobile
base's IP with ``session.verify = False``. This module serves the same routes
from a constant-velocity motion model so the real ROS 2 node can run unmodified
against the simulator.

Three response shapes are dictated by the upstream client and must not change:

* ``/state`` returns a bare JSON string -- ``SpineController`` only detects a
  fault mid-motion when the payload ``isinstance(..., str)``.
* ``/position-mm`` returns integer millimetres -- the controller declares a
  motion finished after three consecutive identical samples at 10 Hz.
* ``motion-mm:start`` returns an object carrying ``StopBy``.

Stdlib only: no web framework and no new runtime dependency. TLS is enabled by
supplying a certificate; ``main()`` generates a self-signed one via the
``openssl`` CLI when none is given.
"""

import argparse
import json
import logging
import math
import ssl
import subprocess
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Tuple

logger = logging.getLogger(__name__)

STATE_SWITCHED_OFF = "SwitchedOff"
STATE_SWITCHED_ON = "SwitchedOn"
STATE_FAULT = "Fault"

#: States upstream's SpineController treats as faulted.
FAULT_STATES = (STATE_FAULT, "FaultReactionActive")

#: Base path of the REST API (SpineApiClient uses https://<ip>/spine/api).
API_PREFIX = "/spine/api"

#: Pinned loopback alias for the spine device (base .10, left .11, right .12).
SPINE_DEFAULT_HOST = "127.0.0.13"

#: SpineApiClient hardcodes ``https://<ip>/spine/api`` with no port, so the
#: device is always on 443. See the plan's privilege note for binding it.
SPINE_DEFAULT_PORT = 443

#: Where main() caches its generated self-signed certificate.
SPINE_CERT_DIR = Path.home() / ".cache" / "franka_sim"


class SpineError(Exception):
    """A refused spine operation, carrying the HTTP status to report."""

    def __init__(self, message: str, status: int = 400):
        super().__init__(message)
        self.status = status
        self.message = message


class SpineModel:
    """Spine state machine with a constant-velocity motion profile.

    ``clock`` is injectable so tests can advance time deterministically; the
    default is :func:`time.monotonic`.
    """

    #: Travel limits, matching franka_spine_vertical_joint in the duo URDF
    #: (``<limit lower="0.0" upper="0.85"/>``).
    LOWER_LIMIT_M = 0.0
    UPPER_LIMIT_M = 0.85

    def __init__(self, position_m: float = 0.0, clock: Callable[[], float] = time.monotonic):
        self._clock = clock
        self.state = STATE_SWITCHED_OFF
        self._start_position = float(position_m)
        self._target_position = float(position_m)
        self._velocity = 0.0
        self._start_time = clock()
        self._lock = threading.Lock()

    def position_m(self) -> float:
        """Current carriage position in metres."""
        with self._lock:
            return self._position_locked()

    def _position_locked(self) -> float:
        if self._velocity <= 0.0:
            return self._target_position
        travelled = self._velocity * (self._clock() - self._start_time)
        distance = self._target_position - self._start_position
        if abs(distance) <= travelled:
            return self._target_position
        return self._start_position + math.copysign(travelled, distance)

    def is_moving(self) -> bool:
        """True while the carriage has not yet reached its target."""
        with self._lock:
            return self._velocity > 0.0 and self._position_locked() != self._target_position

    def switch_on(self) -> str:
        """Enable motion. Refused while faulted."""
        with self._lock:
            if self.state in FAULT_STATES:
                raise SpineError("spine is faulted; reset the fault first", status=409)
            self.state = STATE_SWITCHED_ON
            return self.state

    def switch_off(self) -> str:
        """Stop and disable motion."""
        with self._lock:
            self._freeze_locked()
            self.state = STATE_SWITCHED_OFF
            return self.state

    def fault_reset(self) -> str:
        """Clear a fault; a no-op in any other state."""
        with self._lock:
            if self.state in FAULT_STATES:
                self.state = STATE_SWITCHED_OFF
            return self.state

    def halt(self) -> str:
        """Stop the carriage where it is and report the current state."""
        with self._lock:
            self._freeze_locked()
            return self.state

    def trigger_fault(self) -> str:
        """Force the device into a fault (used by tests and manual scenarios)."""
        with self._lock:
            self._freeze_locked()
            self.state = STATE_FAULT
            return self.state

    def start_motion(self, position_m: float, velocity_mps: float) -> None:
        """Begin an absolute move to ``position_m`` at ``velocity_mps``."""
        with self._lock:
            if self.state != STATE_SWITCHED_ON:
                raise SpineError(f"spine is {self.state}, not {STATE_SWITCHED_ON}", status=409)
            if not self.LOWER_LIMIT_M <= position_m <= self.UPPER_LIMIT_M:
                raise SpineError(
                    f"target {position_m} m is outside "
                    f"[{self.LOWER_LIMIT_M}, {self.UPPER_LIMIT_M}] m",
                    status=400,
                )
            if velocity_mps <= 0.0:
                raise SpineError("velocity must be positive", status=400)

            self._start_position = self._position_locked()
            self._target_position = float(position_m)
            self._velocity = float(velocity_mps)
            self._start_time = self._clock()

    def parameters(self) -> Dict[str, Any]:
        """User limits in the millimetre form SpineApiClient expects."""
        return {
            "user_limits": {
                "lower_limit_in_mm": int(round(self.LOWER_LIMIT_M * 1000)),
                "upper_limit_in_mm": int(round(self.UPPER_LIMIT_M * 1000)),
            }
        }

    def _freeze_locked(self) -> None:
        position = self._position_locked()
        self._start_position = position
        self._target_position = position
        self._velocity = 0.0


class _SpineRequestHandler(BaseHTTPRequestHandler):
    """Route table for the spine REST API."""

    protocol_version = "HTTP/1.1"

    @property
    def model(self) -> SpineModel:
        return self.server.spine_model

    def log_message(self, fmt, *args):  # noqa: D102 - silence the stdlib access log
        logger.debug("spine-stub %s", fmt % args)

    def _send_json(self, payload: Any, status: int = 200) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _endpoint(self) -> Optional[str]:
        path = self.path.split("?", 1)[0]
        if not path.startswith(API_PREFIX + "/"):
            return None
        return path[len(API_PREFIX) + 1 :]

    def _read_json(self) -> Dict[str, Any]:
        length = int(self.headers.get("Content-Length") or 0)
        if length == 0:
            return {}
        try:
            return json.loads(self.rfile.read(length))
        except json.JSONDecodeError as error:
            raise SpineError(f"invalid JSON body: {error}", status=400)

    def do_GET(self):  # noqa: N802 - stdlib naming
        """Handle the read-only endpoints."""
        endpoint = self._endpoint()
        try:
            if endpoint == "state":
                self._send_json(self.model.state)
            elif endpoint == "position-mm":
                self._send_json({"position": int(round(self.model.position_m() * 1000))})
            elif endpoint == "parameters":
                self._send_json(self.model.parameters())
            else:
                self._send_json({"error": "not found"}, status=404)
        except SpineError as error:
            self._send_json({"error": error.message}, status=error.status)

    def do_POST(self):  # noqa: N802 - stdlib naming
        """Handle the command endpoints."""
        endpoint = self._endpoint()
        try:
            if endpoint == "spine:switch-on":
                self._send_json(self.model.switch_on())
            elif endpoint == "spine:switch-off":
                self._send_json(self.model.switch_off())
            elif endpoint == "spine:fault-reset":
                self._send_json(self.model.fault_reset())
            elif endpoint == "motion:halt":
                self._send_json(self.model.halt())
            elif endpoint == "motion-mm:start":
                body = self._read_json()
                self.model.start_motion(
                    float(body.get("position", 0)) / 1000.0,
                    float(body.get("velocity", 0)) / 1000.0,
                )
                self._send_json({"StopBy": "TargetReached"})
            else:
                self._send_json({"error": "not found"}, status=404)
        except SpineError as error:
            self._send_json({"error": error.message}, status=error.status)


class SpineStubServer:
    """Serve :class:`SpineModel` over HTTP, or HTTPS when a certificate is given."""

    def __init__(
        self,
        host: str = "0.0.0.0",
        port: int = 443,
        model: Optional[SpineModel] = None,
        certfile=None,
        keyfile=None,
    ):
        self.model = model if model is not None else SpineModel()
        self._httpd = ThreadingHTTPServer((host, port), _SpineRequestHandler)
        self._httpd.spine_model = self.model
        self._thread: Optional[threading.Thread] = None

        if certfile is not None:
            context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
            context.load_cert_chain(certfile=str(certfile), keyfile=str(keyfile))
            self._httpd.socket = context.wrap_socket(self._httpd.socket, server_side=True)
            logger.info("Spine stub serving HTTPS with %s", certfile)

    @property
    def port(self) -> int:
        """TCP port bound by the server; resolves an ephemeral 0 to its real value."""
        return self._httpd.server_address[1]

    def start(self) -> None:
        """Serve in a daemon thread."""
        self._thread = threading.Thread(
            target=self._httpd.serve_forever, name="spine-stub", daemon=True
        )
        self._thread.start()

    def serve_forever(self) -> None:
        """Serve in the calling thread (blocks)."""
        self._httpd.serve_forever()

    def stop(self) -> None:
        """Shut the listener down and join the serving thread."""
        self._httpd.shutdown()
        self._httpd.server_close()
        if self._thread is not None:
            self._thread.join(timeout=3.0)
            self._thread = None


def make_self_signed_cert(directory) -> Tuple[Path, Path]:
    """Generate a throwaway self-signed certificate with the ``openssl`` CLI.

    ``franka_spine_server`` sets ``verify=False``, so an untrusted certificate
    is exactly what the real device presents from the client's point of view.
    """
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    certfile = directory / "spine-stub-cert.pem"
    keyfile = directory / "spine-stub-key.pem"
    if certfile.exists() and keyfile.exists():
        return certfile, keyfile

    subprocess.run(
        [
            "openssl",
            "req",
            "-x509",
            "-newkey",
            "rsa:2048",
            "-nodes",
            "-days",
            "365",
            "-subj",
            "/CN=franka-spine-stub",
            "-keyout",
            str(keyfile),
            "-out",
            str(certfile),
        ],
        check=True,
        capture_output=True,
    )
    return certfile, keyfile


def main():
    """Run the fake spine REST device."""
    parser = argparse.ArgumentParser(description="Run a fake Franka Spine REST server")
    parser.add_argument(
        "--host",
        default=SPINE_DEFAULT_HOST,
        help=f"Address to bind (default: {SPINE_DEFAULT_HOST})",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=SPINE_DEFAULT_PORT,
        help=f"TCP port (default: {SPINE_DEFAULT_PORT}, the real device's port)",
    )
    parser.add_argument("--cert", default=None, help="TLS certificate (PEM)")
    parser.add_argument("--key", default=None, help="TLS private key (PEM)")
    parser.add_argument("--no-tls", action="store_true", help="Serve plain HTTP (debugging only)")
    parser.add_argument(
        "--initial-position",
        type=float,
        default=0.0,
        help="Initial carriage position in metres (default: 0.0)",
    )
    args = parser.parse_args()

    certfile = args.cert
    keyfile = args.key
    if not args.no_tls and certfile is None:
        certfile, keyfile = make_self_signed_cert(SPINE_CERT_DIR)

    server = SpineStubServer(
        host=args.host,
        port=args.port,
        model=SpineModel(position_m=args.initial_position),
        certfile=certfile,
        keyfile=keyfile,
    )
    scheme = "http" if args.no_tls else "https"
    print(f"Spine stub listening on {scheme}://{args.host}:{server.port}{API_PREFIX}")
    print("Press Ctrl+C to stop the server")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down spine stub...")
        server.stop()


if __name__ == "__main__":
    main()
