from math import atan2, pi

import gym
import numpy
import roboschool.gym_pendulums as g

from controller import PIDController


def main():
    type_g = str(type(g))
    env = gym.make("RoboschoolInvertedPendulum-v1")
    pid_controller = PIDController(50, 0.0, 10, 0.0165)
    pid_offset_controller = PIDController(10.0, 0.0, 5, 0.0165)
    observation = env.reset()
    action = numpy.array([0])
    for _ in range(10000):
        env.render()
        observation, reward, done, info = env.step(action)
        theta = atan2(observation[3], observation[2])
        sin_theta = observation[3]
        cos_theta = observation[2]
        offset = observation[0]
        angle_error = theta if theta <= pi else 2 * pi - theta
        sum_action = pid_controller.step(angle_error)
        offset_controller = pid_offset_controller.step(offset)

        if angle_error * offset > 0:
            sum_action += 0
        else:
            sum_action += offset_controller

        action[0] = sum_action
        if done:
            pass
            # observation = env.reset()
    env.close()


if __name__ == '__main__':
    main()
