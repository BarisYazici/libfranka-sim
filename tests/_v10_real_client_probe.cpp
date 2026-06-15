// Minimal real libfranka_new (v10) client used by test_real_client_integration.py
// to prove wire compatibility against the simulator.
//
// It constructs a franka::Robot with RealtimeConfig::kIgnore (so it runs on a
// non-realtime kernel), which exercises the full v10 handshake:
//   connect -> read one RobotState (1377 bytes) -> getRobotModel (URDF).
// It then reads a few more states and prints machine-checkable markers.
#include <franka/exception.h>
#include <franka/robot.h>

#include <iostream>
#include <string>

int main(int argc, char** argv) {
  if (argc != 2) {
    std::cerr << "usage: probe <robot-ip>\n";
    return 2;
  }
  try {
    franka::Robot robot(argv[1], franka::RealtimeConfig::kIgnore);
    std::string urdf = robot.getRobotModel();
    std::cout << "CONNECT_OK getRobotModel_bytes=" << urdf.size()
              << " has_joint1=" << (urdf.find("\"joint1\"") != std::string::npos) << std::endl;

    size_t count = 0;
    robot.read([&count](const franka::RobotState&) { return ++count < 5; });
    std::cout << "READ_OK states_read=" << count << std::endl;
  } catch (const franka::Exception& e) {
    std::cout << "EXCEPTION: " << e.what() << std::endl;
    return 1;
  }
  return 0;
}
