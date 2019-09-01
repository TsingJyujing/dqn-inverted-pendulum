from math import atan2, pi, ceil

import gym
import numpy

from controller import PIDController
from model import RoboschoolInvertedPendulum


def main():
    print(str(type(RoboschoolInvertedPendulum)))
    dt = 0.0165
    simulate_time = 10  # second
    env = gym.make("TsingJyujingInvertedPendulum-v1")
    pid_controller = PIDController(10, 0.0, 10, dt)
    pid_offset_controller = PIDController(1.0, 0.0, 8, dt)
    observation = env.reset()
    action = numpy.array([0])
    for _ in range(ceil(simulate_time / dt)):
        env.render()
        observation, reward, done, info = env.step(action)
        theta = atan2(observation[3], observation[2])
        sin_theta = observation[3]
        cos_theta = observation[2]
        omega = observation[4]
        position_error = observation[0]
        angle_error = theta if theta <= pi else 2 * pi - theta
        angle_control_value = pid_controller.step(angle_error)
        position_control_value = pid_offset_controller.step(position_error)
        sum_action = angle_control_value + position_control_value
        action[0] = sum_action
    env.close()


if __name__ == '__main__':
    main()
