import numpy as np

from toy_gym.envs.toy_tasks.toy_pickplace2D import ToyPickPlace2D
from toy_gym.pybullet_gym import PyBullet

class ToyPickPlaceFiveObject(ToyPickPlace2D):
    def __init__(self, render: bool = False, 
                    object_positions = [[-1.0, -1.0], [3.0,3.0] , [3.0,2.0], [-2.0,3.0], [2.0,-4.0]],
                    target_positions = [[2.0,4.0], [-2.5, 4.0], [-1.0, 4.0], [0.0, 4.0], [1.0, 4.0]],
                    is_target_random = False,
                    is_object_random = False,
                    map_name="maze_world",
                    verbose = False) -> None:
        super().__init__(render=render, is_target_random=is_target_random, is_object_random = is_object_random, no_of_random_object=5,
                    object_positions=object_positions, target_positions=target_positions, map_name=map_name, verbose=verbose)
        
