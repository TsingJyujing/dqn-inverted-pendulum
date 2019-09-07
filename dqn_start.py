import datetime
import json
import os
import random
import sys
import traceback
from math import atan2, pi, ceil

import click
import gym
import numpy
import torch

from controller import DQNController

try:
    from model import RoboschoolInvertedPendulum, FirstOrderReward
except:
    print("Error while loading the model.")


@click.command(name="run_dqn")
@click.option("--greedy-lambda", default="lambda t:random.random() < 0.1", help="The prob of greedy randomly")
@click.option("--render/--no-render", default=False, help="Is render the environment")
@click.option("--train/--no-train", default=False, help="Is training the network")
@click.option("--load-model", default="", help="Is trying to load pre-trained model")
@click.option("--data-log", help="Where to save data log")
@click.option("--simulate-time", default=4000.0, help="How many seconds to simulate")
@click.option("--training-gap-time", default=10.0, help="How many seconds gap to train")
@click.option("--power-factor", default=0.01, help="Power penalty factor in reward")
@click.option("--offset-factor", default=0.5, help="Offset penalty factor in reward")
@click.option("--omega-factor", default=0.02, help="Rotate speed penalty factor in reward")
@click.option("--reward-decay", default=0.8, help="Reward future decay factor")
@click.option("--random-seconds", default=0.0, help="Random move initial steps")
@click.option("--training-time-range", default=800.0, help="How many seconds using for training")
@click.option("--training-max-epochs", default=10000, help="While training, how many steps to iter")
@click.option("--training-early-stopping-epochs", default=30, help="While training, how many steps to iter")
def main(
        greedy_lambda: str,
        render: bool,
        train: bool,
        load_model: str,
        data_log: str,
        simulate_time: float,
        training_gap_time: float,
        power_factor: float,
        offset_factor: float,
        omega_factor: float,
        reward_decay: float,
        random_seconds: float,
        training_time_range: float,
        training_max_epochs: int,
        training_early_stopping_epochs: int,

):
    data_tag = datetime.datetime.now().strftime("DQN-IP1-%Y%m%d-%H%M%S")
    data_dir = os.path.join(data_log, data_tag)
    if not os.path.exists(data_dir):
        os.mkdir(data_dir)
        with open(os.path.join(data_dir, "settings.json"), "w") as fp:
            json.dump({name: value for name, value in locals().items() if type(value) in {
                str, float, int, list, dict, bool
            }}, fp)
        with open(os.path.join(data_dir, "command.txt"), "w") as fp:
            fp.write(" ".join(sys.argv))
        model_log_dir = os.path.join(data_dir, "model")
        os.mkdir(model_log_dir)
    else:
        raise Exception("Data dir {} is occupied!".format(data_dir))

    dt = 1.0 / 60.0
    train_gap_step = ceil(training_gap_time / dt)
    greedy = eval(greedy_lambda)

    controller = DQNController(
        reward=FirstOrderReward(
            dt=dt,
            power_factor=power_factor,
            offset_factor=offset_factor,
            omega_factor=omega_factor,
            r_decay=reward_decay
        ),
        greedy_epsilon=greedy,
        force_set=numpy.linspace(-10, 10, 50).tolist(),
        training_time=training_time_range,
    )

    if len(load_model) > 0:
        try:
            with open(load_model, "rb") as fp:
                controller.set_model(torch.load(fp))
        except:
            print("Load model failed, using init model: \n{}".format(traceback.format_exc()))

    env = gym.make("TsingJyujingInvertedPendulum-v1")
    _ = env.reset()
    action = numpy.array([0])
    angle_error_sum = 0
    with open(os.path.join(data_dir, "status.csv"), "w") as fp:
        fp.write("tick,offset,angle,velocity,omega,input_force,decided_force\n")

        for i in range(ceil(simulate_time / dt)):
            tick = i * dt
            if render:
                env.render()
            observation, reward, done, info = env.step(action)
            theta = atan2(observation[3], observation[2])
            omega = observation[4]
            vel = observation[1]
            position_error = observation[0]
            angle_error = theta if theta <= pi else 2 * pi - theta
            F = controller.step(tick, offset=position_error, velocity=vel, theta=angle_error, omega=omega)
            if tick <= random_seconds:
                F = random.gauss(0, 3)
            fp.write("{},{},{},{},{},{},{}\n".format(
                tick,
                position_error,
                angle_error,
                vel,
                omega,
                action[0],
                F
            ))
            action[0] = F
            angle_error_sum += abs(angle_error)
            if train and i % train_gap_step == (train_gap_step - 1):
                print("\n  Average angle error: {}".format(angle_error_sum / train_gap_step))
                angle_error_sum = 0
                controller.train(
                    steps=training_max_epochs,
                    early_stop_move=training_early_stopping_epochs
                )
                with open(os.path.join(model_log_dir, "step-{}.pkl".format(i)), "wb") as model_fp:
                    torch.save(controller.model, model_fp)
    with open(os.path.join(data_dir, "model.pkl"), "wb") as model_fp:
        torch.save(controller.model, model_fp)
    env.close()


if __name__ == '__main__':
    main()
