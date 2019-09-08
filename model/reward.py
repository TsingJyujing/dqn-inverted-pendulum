from abc import abstractmethod
from math import log, exp, pi
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

    def update(self, tick: float, stat: numpy.ndarray) -> Tuple[float, float, numpy.ndarray]:
        inst_q = self.instantaneous_reward(stat)
        self.reward_buffer.append((tick, inst_q, stat))
        if len(self.reward_buffer) > self.future_step:
            p_tick, p_Q, p_stat = self.reward_buffer.pop(0)
            sum_weight = 0.0
            sum_weighted_reward = 0.0
            for t, r, _ in self.reward_buffer:
                w = self.get_decay(abs(p_tick - t))
                sum_weight += w
                sum_weighted_reward += w * r
            Q = sum_weighted_reward / sum_weight
            return Q, inst_q, p_stat
        else:
            return inst_q, inst_q, stat


class FirstOrderReward(Reward):

    def __init__(
            self,
            dt: float,
            power_factor: float,
            offset_factor: float,
            omega_factor: float,
            r_decay: float,
            effect_limit: float = 0.05
    ):
        super().__init__(dt, r_decay, effect_limit)
        self.power_factor = power_factor
        self.offset_factor = offset_factor
        self.omega_factor = omega_factor

    def instantaneous_reward(self, stat: numpy.ndarray) -> float:
        """
                    0        1        2       3     4
        :param stat: offset, velocity, theta, omega, force
        :return:
        """
        theta = stat[2]
        instr = exp(abs(theta)) - 1
        if abs(theta) > 0.7:
            instr += self.offset_factor * (exp(pi) - 1)
        else:
            instr += self.offset_factor * (exp(abs(stat[0])) - 1)
        instr += self.power_factor * abs(stat[1] * stat[4])

        instr += self.omega_factor * stat[3] * stat[3]
        return -instr
