# Franka Simulation Server

A high-fidelity Genesis simulation server that communicates with the Franka robot's network protocol, enabling seamless switching between simulation and hardware.

## Overview

The Franka Simulation Server provides a drop-in replacement for the real Franka robot, implementing the complete libfranka network protocol. This allows developers to:

- Test and debug robot controllers in simulation before deployment
- Develop applications that work identically on both simulation and hardware
- Validate error handling and safety features
- Experiment with different control strategies risk-free

## Compatibility

`franka_sim` implements the **latest libfranka wire protocol — robot server version 10**.
In this protocol the robot model is built **client-side**: during connection the client
fetches the robot URDF via the `GetRobotModel` command and builds its own Pinocchio model.
The server therefore just serves a valid URDF (a hand-less FR3 arm by default) instead of a
precompiled model library.

You need a libfranka version that speaks **server protocol 10**:

| libfranka Version | Robot System Version | Robot/Gripper Server |
| ----------------- | -------------------- | -------------------- |
| **>= 0.18.0**     | **>= 5.9.0**         | **10 / 3** ✅ supported |
| >= 0.15.0         | >= 5.7.2             | 9 / 3 — not supported |

See the [Franka software compatibility matrix](https://frankarobotics.github.io/docs/compatibility.html)
for the full list. Older libfranka releases (server version 9 and below) use a different wire
format — double-based `RobotState` and server-side `LoadModelLibrary` instead of `GetRobotModel` —
and are **not** compatible with this server.

## Related Projects

- [libfranka-python](https://github.com/BarisYazici/libfranka-python) - Python bindings for libfranka
- [franka-gym](https://github.com/BarisYazici/franka-gym) - Franka gym implementation

## Preview Video

- Native libfranka control

![Libfranka Native Control](./assets/direct_libfranka_control.gif)

- With Python

![With Python](./assets/libfranka_python_bindings_control.gif)


## Architecture

![Architecture](./assets/libfranka_sim.svg)

In this repository, we only provide the simulation server backend with Genesis connection.

The libfranka python bindings will become available in a separate repository.

The system consists of several key components:

1. **libfranka Interface Layer**
   - Implements the standard Franka robot network protocol
   - Handles TCP command interface and UDP state updates
   - Targets the latest libfranka wire protocol (robot server version 10)

2. **Genesis Simulation Backend**
   - Physics-based robot simulation using the Genesis engine
   - Real-time joint state computation and dynamics

3. **State Management**
   - Complete robot state tracking and synchronization
   - Accurate error reporting and status updates
   - Real-time state transmission (1kHz update rate)

4. **Control Modes**
   - Joint Position Control
   - Joint Velocity Control
   - Joint Torque Control
   - Supports seamless switching between modes

## Key Features

- **Protocol Compatibility**: Full implementation of the Franka robot network protocol
- **Real-time Simulation**: High-frequency state updates and control (1kHz)
- **Multiple Control Modes**: Supports position, velocity, and torque control
- **Error Handling**: Replicates real robot error states and recovery

## Getting Started

### Prerequisites
- Python 3.9+
- genesis-world==0.2.1
- numpy==1.26.4
- numba==0.60.0

### Installation

#### Option 1: Install from PyPI (Recommended)

The package is available on PyPI and can be installed with pip:

```bash
pip install franka-sim
```

#### Option 2: Install from Source

```bash
# Clone the repository
git clone git@github.com:BarisYazici/libfranka-sim.git

# Install the package
cd libfranka-sim/simulation
pip install -e .
```

### Basic Usage

After installation, you can run the server using the command-line executable:

```bash
# Start the server without visualization
run-franka-sim-server

# Start the server with visualization
run-franka-sim-server -v
```

Alternatively, if you installed from source, you can use:

```bash
# Start the simulation server
python -m franka_sim.run_server -v
```

In your application, use standard libfranka commands. The simulation will respond exactly like the real robot.

### Gripper (Franka Hand)

The gripper server (libfranka gripper protocol, **TCP port 1338**) runs **by default** alongside
the arm. Drive it with the standard `franka::Gripper` client (`homing` / `move` / `grasp` / `stop`).

```bash
# Arm + gripper, kinematic hand (no mesh; width tracked analytically, CI-friendly)
python -m franka_sim.run_server -v

# Arm + gripper, physics-backed Franka Hand: the hand mesh is loaded and the
# finger DOFs are simulated, so homing/move/grasp visibly move the fingers in
# the viewer (grasp succeeds on a finger-position stall against an object)
python -m franka_sim.run_server -v --gripper-physics

# Disable the gripper server entirely
python -m franka_sim.run_server -v --no-gripper
```

| Flag | Gripper backend | Hand in viewer |
| ---- | --------------- | -------------- |
| *(default)* | `FrankaHandSim` (kinematic) | no |
| `--gripper-physics` | `GenesisFrankaHand` (physics) | yes, fingers move |
| `--no-gripper` | none (arm only) | no |

### Troubleshooting

If you encounter issues related to missing asset files, make sure you have the correct version of `genesis-world` installed:

```bash
pip install genesis-world==0.2.1
```

The simulator now automatically uses the assets provided by the Genesis package, so no additional asset files are needed.

## Configuration

## Switching Between Simulation and Hardware

To switch between simulation and hardware:

1. Update the robot IP address in your application:
   - Use `localhost` or `127.0.0.1` for simulation
   - Use the real robot's IP for hardware

2. No other changes needed - your application code remains identical

## Development Status

The simulation server currently implements all major features of the Franka robot:

- [x] Complete network protocol implementation
- [x] All joint interfaces
- [x] Real-time state updates
- [x] Visualization support
- [x] Genesis connection
- [x] libfranka python bindings
- [x] v10 wire protocol (Connect, float-based RobotState, GetRobotModel/URDF)
- [x] Robot model via URDF (client-side Pinocchio through GetRobotModel)
- [x] Gripper simulation / Franka Hand (kinematic + Genesis physics, `--gripper-physics`)
- [x] Automatic error recovery (so `franka_hardware` / franka_ros2 can activate)
- [ ] Advanced collision detection (in progress)
- [ ] Cartesian interfaces (planned)



## Contributing

Contributions are welcome! Please read our contributing guidelines and submit pull requests to our repository.

## License

This project is licensed under the Apache License Version 2.0 - see the LICENSE file for details.

## Acknowledgments

- Franka Robotics GmbH for the original libfranka implementation
- The Genesis Simulator team for the physics engine
