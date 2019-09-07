import random
from math import ceil
from typing import List, Iterable, Callable

import numpy
import torch
import torch.nn
import torch.nn.functional

from model import Reward


class DQN(torch.nn.Module):
    def __init__(self, layer_size: List[int]):
        super().__init__()
        self.layers = torch.nn.ModuleList(modules=[
            torch.nn.Linear(input_size, output_size)
            for input_size, output_size in zip(
                layer_size[:-2],
                layer_size[1:-1]
            )
        ])
        self.final_layer = torch.nn.Linear(layer_size[-2], layer_size[-1])

    def forward(self, x):
        for l in self.layers:
            x = torch.relu(l(x))
        return self.final_layer(x)


class DQNController:
    def __init__(
            self,
            reward: Reward,
            greedy_epsilon: Callable[[float], bool],
            force_set: Iterable[float],
            training_time: float,
            dqn_layers: List[int] = None

    ):
        if dqn_layers is None:
            dqn_layers = [5, 32, 32, 1]
        self.greedy_epsilon = greedy_epsilon
        self.reward = reward
        self.training_data = []
        self.training_data_limit = ceil(training_time / reward.dt)
        self.set_model(DQN(dqn_layers))
        self.force_set = torch.FloatTensor(force_set)

    def set_model(self, new_model):
        self.model = new_model
        self.optimizor = torch.optim.Adam(
            self.model.parameters(),
            lr=0.01
        )

    def step(self, tick: float, offset: float, velocity: float, theta: float, omega: float) -> float:
        violent_input = torch.zeros(size=(self.force_set.shape[0], 5), dtype=torch.float32)
        violent_input[:, 0] = offset
        violent_input[:, 1] = velocity
        violent_input[:, 2] = theta
        violent_input[:, 3] = omega
        violent_input[:, 4] = self.force_set
        if self.greedy_epsilon(tick):
            F = (random.random() - 0.5) * 10
        else:
            result = self.model(violent_input)
            F = self.force_set.numpy()[int(torch.argmax(result).numpy())]
        status_str = "x={} v={} theta={} omega={} F={}".format(
            offset,
            velocity,
            theta,
            omega,
            F
        )
        stats = numpy.array([offset, velocity, theta, omega, F])
        Q, inst_q, p_stat = self.reward.update(tick, stats)
        self.training_data.append((Q, p_stat))
        if len(self.training_data) > self.training_data_limit:
            self.training_data.pop(0)
        print("\rTick={} Q={} instQ={} status: {}   ".format(tick, Q, inst_q, status_str), end="")
        return F

    def train(self, steps: int = 10000, early_stop_move: int = 10):
        print("")

        X_raw = torch.FloatTensor([stat for _, stat in self.training_data])
        y_raw = torch.FloatTensor([Q for Q, _ in self.training_data])
        index_list = list(range(len(self.training_data)))
        random.shuffle(index_list)
        split_point = ceil(len(index_list) * 0.2)
        X_test = X_raw[:split_point, :]
        y_test = y_raw[:split_point]
        X = X_raw[split_point:, :]
        y = y_raw[split_point:]
        min_loss = 10000000000
        min_loss_index = -1
        for totle_move in range(steps):
            self.model.train(True)
            self.optimizor.zero_grad()
            pred = self.model(X)
            loss = torch.nn.functional.mse_loss(pred[:, 0], y)
            loss.backward()
            self.optimizor.step()
            self.model.train(False)
            test_pred = self.model(X_test)
            test_loss = torch.nn.functional.mse_loss(test_pred[:, 0], y_test)
            test_loss_value = test_loss.detach().numpy()
            print("\rStep={} Loss={} TestLoss={}, EarlyStop={}".format(
                totle_move,
                loss.detach().numpy(),
                test_loss_value,
                (totle_move - min_loss_index)
            ), end="")
            if test_loss_value <= min_loss:
                min_loss_index = totle_move
                min_loss = test_loss_value

            if (totle_move - min_loss_index) >= early_stop_move:
                break

        print("")
