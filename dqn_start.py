import random
import traceback
from math import atan2, pi, ceil

import gym
import numpy
import torch

from controller import DQNController
from model import RoboschoolInvertedPendulum, FirstOrderReward


def main():
    print(str(type(RoboschoolInvertedPendulum)))

    dt = 0.0165
    simulate_time = 10
    greedy = lambda t: random.random() < 0.1
    model_file = "dqn.pkl"

    controller = DQNController(
        reward=FirstOrderReward(
            dt=dt,
            power_factor=0.1,
            offset_factor=0.1,
            r_decay=2.0
        ),
        greedy_epsilon=greedy,
        force_set=numpy.linspace(-10, 10, 50).tolist(),
        training_time=600,
    )

    try:
        with open(model_file, "rb") as fp:
            controller.model = torch.load(fp)
    except:
        print("Load model failed, using init model: \n{}".format(traceback.format_exc()))
    env = gym.make("TsingJyujingInvertedPendulum-v1")
    _ = env.reset()
    action = numpy.array([0])
    for i in range(ceil(simulate_time / dt)):
        tick = i * dt
        env.render()
        observation, reward, done, info = env.step(action)
        theta = atan2(observation[3], observation[2])
        # sin_theta = observation[3]
        # cos_theta = observation[2]
        # omega = observation[4]
        vel = observation[1]
        position_error = observation[0]
        angle_error = theta if theta <= pi else 2 * pi - theta
        action[0] = controller.step(tick, offset=position_error, velocity=vel, theta=angle_error)
        if i % 1000 == 199:
            controller.train()
    env.close()
    with open(model_file, "wb") as fp:
        torch.save(controller.model, fp)


if __name__ == '__main__':
    main()
