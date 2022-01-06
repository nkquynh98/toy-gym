from toy_gym.envs.toy_tasks.toy_pickplace_oneobject import ToyPickPlaceOneObject
from toy_gym.envs.toy_tasks.toy_pickplace_fiveobject import ToyPickPlaceFiveObject
from toy_gym.envs.tasks.pick_and_place_2d import PickAndPlace2D
import gym
from numpngw import write_apng
import matplotlib.pyplot as plt
from toy_gym.policy.crude_policy import crude_policy
import pybullet as p


# images = []
# #env = gym.make("ToyPickPlaceOneObject-v1",render=True)
# #env = ToyPickPlaceOneObject(render=True)
env = ToyPickPlaceFiveObject(render=True, map_name="maze_world")
obs = env.reset()
env.task.get_fixed_obstacle()
env.task.get_movable_obstacle()
# policy = crude_policy(env, False)
# done = False