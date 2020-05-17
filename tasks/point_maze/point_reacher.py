from .environments.point_maze_env import PointMazeEnv


def create_point_reacher_env():
    gym_mujoco_kwargs = {
        'maze_id': 'Reacher',
        'n_bins': 0,
        'observe_blocks': False,
        'put_spin_near_agent': False,
        'top_down_view': False,
        'manual_collision': True,
        'maze_size_scaling': 3,
        'color_str': ""
    }
    env = PointMazeEnv(**gym_mujoco_kwargs)

    return env
