from typing import Optional


class PIDController:

    def __init__(self, kp: float, ki: float, kd: float, default_dt: float):
        self.__default_dt = default_dt
        assert default_dt > 0, "Default dt should be greater than zero"
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.integral_value: float = 0.0
        self.last_value: Optional[float] = None

    def get_dt(self, dt: float):
        return dt if dt is not None and dt > 0 else self.__default_dt

    def step(self, error_value: float, error_value_diff: float = None, dt: float = None):
        dif = (error_value - self.last_value) / self.get_dt(dt) if self.last_value is not None else 0.0
        self.last_value = error_value
        self.integral_value += error_value * self.get_dt(dt)
        return self.kp * error_value + self.ki * self.integral_value + self.kd * dif

    def reset(self):
        self.integral_value = 0
        self.last_value = None
