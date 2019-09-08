import os
import random
import re
from math import ceil, atan2, pi

import click
import gym
import numpy
import pandas
import torch

from controller import DQNController
from model import FirstOrderReward


@click.command()
@click.option("--data-log", help="Where to save data log")
@click.option("--render/--no-render", default=False, help="Is render the environment")
@click.option("--noise-level", default=0.0, help="Noise level while evaluating")
@click.option("--test-count", default=1, help="How many times to test")
@click.option("--single-test-time", default=10, help="How long for one test")
@click.option("--freedom-time", default=2, help="How long to keep model free")
def main(
        data_log: str,
        render: bool,
        noise_level: float,
        test_count: int,
        single_test_time: float,
        freedom_time: float
):
    result_data = []
    dt = 1 / 60.0
    controller = DQNController(
        reward=FirstOrderReward(
            dt=dt,
            power_factor=0,
            offset_factor=0,
            omega_factor=0,
            r_decay=0.8
        ),
        greedy_epsilon=lambda t: False,
        force_set=numpy.linspace(-10, 10, 50).tolist(),
        training_time=1,
        noise_level=noise_level
    )
    try:
        with open(os.path.join(data_log, "model.pkl"), "rb") as fp:
            controller.set_model(torch.load(fp))
    except:
        mid = max([int(re.findall("step-(\d+).pkl", f)[0]) for f in os.listdir(os.path.join(data_log, "model")) if
                   f.startswith("step-")])
        with open(os.path.join(data_log, "model", "step-{}.pkl".format(mid)), "rb") as fp:
            controller.set_model(torch.load(fp))
    for test_id in range(test_count):
        env = gym.make("TsingJyujingInvertedPendulum-v1")
        _ = env.reset()
        action = numpy.array([0])
        for i in range(ceil(single_test_time / dt)):
            if render:
                env.render()
            tick = i * dt
            observation, reward, done, info = env.step(action)
            theta = atan2(observation[3], observation[2])
            omega = observation[4]
            vel = observation[1]
            position_error = observation[0]
            angle_error = theta if theta <= pi else 2 * pi - theta
            if tick <= freedom_time:
                F = random.gauss(0, 3)
            else:
                F = controller.step(
                    tick,
                    offset=position_error,
                    velocity=vel,
                    theta=angle_error,
                    omega=omega
                )
            result_data.append(dict(
                iter=i,
                tick=tick,
                offset=position_error,
                velocity=vel,
                theta=angle_error,
                omega=omega,
                apply_F=action[0],
                next_F=F
            ))
            action[0] = F
        env.close()
    df = pandas.DataFrame(result_data)
    df.to_csv("buffer.csv")


if __name__ == '__main__':
    main()
