// Tiny real-client probe: drives the sim's gripper server with franka::Gripper.
// Prints machine-checkable markers the Python test asserts on.
#include <franka/gripper.h>
#include <franka/gripper_state.h>

#include <iostream>

int main(int argc, char** argv) {
  if (argc < 2) {
    std::cerr << "usage: gripper_probe <robot-ip>" << std::endl;
    return 2;
  }
  try {
    franka::Gripper gripper(argv[1]);
    std::cout << "CONNECT_OK" << std::endl;

    bool homed = gripper.homing();
    std::cout << "HOMING=" << (homed ? 1 : 0) << std::endl;

    franka::GripperState state = gripper.readOnce();
    std::cout << "MAX_WIDTH=" << state.max_width << std::endl;
    std::cout << "WIDTH=" << state.width << std::endl;

    bool moved = gripper.move(0.03, 0.1);
    std::cout << "MOVE=" << (moved ? 1 : 0) << std::endl;
  } catch (const std::exception& e) {
    std::cerr << "EXCEPTION: " << e.what() << std::endl;
    return 1;
  }
  return 0;
}
