import numpy as np

from toy_gym.envs.EnvTemplate import RobotTaskEnv
from toy_gym.envs.robots.toy import Toy
from toy_gym.envs.tasks.pick_and_place_2d import PickAndPlace2D
from toy_gym.pybullet_gym import PyBullet

class ToyPickPlaceFiveObject(RobotTaskEnv):
    def __init__(self, render: bool = False, 
                    object_positions = [[-1.0, -1.0], [3.0,3.0] , [3.0,2.0], [-2.0,3.0], [2.0,-4.0]],
                    target_positions = [[2.0,4.0], [-2.5, 4.0], [-1.0, 4.0], [0.0, 4.0], [1.0, 4.0]],
                    map_name="maze_world") -> None:
        self.object_positions = object_positions
        #object_positions = [[-1.0, -1.3], [-1.0, -3] , [-1.0, -1.9], [-1.0, -2.2], [-1.0, -0.70]]
        self.target_positions = target_positions
        sim = PyBullet(render=render,background_color=np.array([0,0,230]))
        robot = Toy(sim)
        task = PickAndPlace2D(sim,object_positions=self.object_positions, target_positions=target_positions, map_name=map_name)
        super().__init__(robot, task)
        
