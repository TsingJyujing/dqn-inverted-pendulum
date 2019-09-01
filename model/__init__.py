from gym.envs.registration import register

register(
    id='TsingJyujingInvertedPendulum-v1',
    entry_point='model:RoboschoolInvertedPendulum',
    max_episode_steps=1000,
    reward_threshold=950.0,
    tags={"pg_complexity": 1 * 1000000},
)
register(
    id='TsingJyujingInvertedDoublePendulum-v1',
    entry_point='model:RoboschoolInvertedDoublePendulum',
    max_episode_steps=1000,
    reward_threshold=9100.0,
    tags={"pg_complexity": 1 * 1000000},
)

from .gym_pendulums import \
    RoboschoolInvertedDoublePendulum, \
    RoboschoolInvertedPendulum
