import numpy as np

from toy_gym.envs.EnvTemplate import RobotTaskEnv
from toy_gym.envs.toy_tasks.toy_pickplace2D import ToyPickPlace2D
from toy_gym.envs.robots.toy import Toy
from toy_gym.envs.tasks.pick_and_place_2d import PickAndPlace2D
from toy_gym.pybullet_gym import PyBullet

class ToyPickPlaceOneObject(RobotTaskEnv):
    def __init__(self, render: bool = False) -> None:
        sim = PyBullet(render=render,background_color=np.array([0,0,230]))
        robot = Toy(sim)
        task = PickAndPlace2D(sim)
        super().__init__(robot, task)
        
