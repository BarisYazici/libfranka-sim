// Real libfranka_new (v10) client used by test_mobile_duo_e2e.py to prove the
// mobile-duo wire contract: cartesian velocity on the base bridge and torque
// control on both arm bridges.
//
// usage: probe <base-ip> <left-ip> <right-ip>
#include <franka/exception.h>
#include <franka/robot.h>

#include <array>
#include <cstddef>
#include <iostream>

int main(int argc, char** argv) {
  if (argc != 4) {
    std::cerr << "usage: probe <base-ip> <left-ip> <right-ip>\n";
    return 2;
  }
  try {
    // --- base: cartesian velocity -> swerve IK -> wheel state -------------
    franka::Robot base(argv[1], franka::RealtimeConfig::kIgnore);
    std::array<double, 7> first{}, last{};
    std::size_t cycles = 0;

    base.control([&](const franka::RobotState& state,
                     franka::Duration) -> franka::CartesianVelocities {
      if (cycles == 0) {
        first = state.q;
      }
      last = state.q;
      franka::CartesianVelocities twist{{0.2, 0.0, 0.0, 0.0, 0.0, 0.0}};
      if (++cycles >= 500) {
        return franka::MotionFinished(twist);
      }
      return twist;
    });

    std::cout << "BASE_OK cycles=" << cycles << " drive_delta=" << (last[1] - first[1])
              << " steer=" << last[0] << std::endl;

    // --- arms: zero-torque external control -------------------------------
    for (int index = 2; index < 4; ++index) {
      franka::Robot arm(argv[index], franka::RealtimeConfig::kIgnore);
      std::size_t arm_cycles = 0;
      arm.control([&](const franka::RobotState&, franka::Duration) -> franka::Torques {
        franka::Torques zero{{0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0}};
        if (++arm_cycles >= 200) {
          return franka::MotionFinished(zero);
        }
        return zero;
      });
      std::cout << "ARM_OK " << argv[index] << " cycles=" << arm_cycles << std::endl;
    }
  } catch (const franka::Exception& e) {
    std::cout << "EXCEPTION: " << e.what() << std::endl;
    return 1;
  }
  return 0;
}
