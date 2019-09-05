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
            training_time: float

    ):
        self.greedy_epsilon = greedy_epsilon
        self.reward = reward
        self.training_data = []
        self.training_data_limit = ceil(training_time / reward.dt)
        self.model = DQN([4, 32, 32, 1])
        self.force_set = torch.FloatTensor(force_set)
        self.optimizor = torch.optim.Adam(
            self.model.parameters(),
            lr=0.01
        )

    def step(self, tick: float, offset: float, velocity: float, theta: float) -> float:
        violent_input = torch.zeros(size=(self.force_set.shape[0], 4), dtype=torch.float32)
        violent_input[:, 0] = offset
        violent_input[:, 1] = velocity
        violent_input[:, 2] = theta
        violent_input[:, 3] = self.force_set
        if self.greedy_epsilon(tick):
            F = (random.random() - 0.5) * 20
        else:
            result = self.model(violent_input)
            F = self.force_set.numpy()[int(torch.argmax(result).numpy())]

        stats = numpy.array([offset, velocity, theta, F])
        Q, inst_q = self.reward.update(tick, stats)
        self.training_data.append((Q, stats))
        if len(self.training_data) > self.training_data_limit:
            self.training_data.pop(0)
        print("Tick={} Q={} instQ={} F={}".format(tick, Q, inst_q, F))
        return F

    def train(self, steps: int):
        self.model.train(True)
        X = torch.FloatTensor([stat for _, stat in self.training_data])
        y = torch.FloatTensor([Q for Q, _ in self.training_data])
        for i in range(steps):
            self.optimizor.zero_grad()
            pred = self.model(X)
            loss = torch.nn.functional.mse_loss(pred[:, 0], y)
            loss.backward()
            self.optimizor.step()
            print("Step={} Loss={}".format(i, loss.detach().numpy()))
        self.model.train(False)
