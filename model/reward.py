from abc import abstractmethod
from math import log, exp
from typing import Tuple

import numpy


class Reward:
    def __init__(
            self,
            dt: float,
            r_decay: float,
            effect_limit: float = 0.05,
    ):
        self.dt = dt
        self.r_decay = r_decay
        self.future_step = -log(effect_limit) / (r_decay * dt)
        self.reward_buffer = []

    def get_decay(self, delta_tick: float):
        return exp(-delta_tick * self.r_decay)

    @abstractmethod
    def instantaneous_reward(self, stat: numpy.ndarray) -> float:
        pass

    def update(self, tick: float, stat: numpy.ndarray) -> Tuple[float, float]:
        inst_q = self.instantaneous_reward(stat)
        self.reward_buffer.append((tick,inst_q))
        if len(self.reward_buffer) > self.future_step:
            self.reward_buffer.pop(0)

        sum_weight = 0.0
        sum_weighted_reward = 0.0
        for t, r in self.reward_buffer:
            w = self.get_decay(abs(tick - t))
            sum_weight += w
            sum_weighted_reward += w * r

        Q = sum_weighted_reward / sum_weight
        return Q, inst_q


class FirstOrderReward(Reward):

    def __init__(self, dt: float, power_factor: float, offset_factor: float, r_decay: float,
                 effect_limit: float = 0.05):
        super().__init__(dt, r_decay, effect_limit)
        self.power_factor = power_factor
        self.offset_factor = offset_factor

    def instantaneous_reward(self, stat: numpy.ndarray) -> float:
        """
                    0        1        2       3
        :param stat: offset, velocity, theta , force
        :return:
        """
        instr = stat[2] * stat[2] + self.power_factor * stat[1] * stat[3] + self.offset_factor * stat[0] * stat[0]
        return -instr
