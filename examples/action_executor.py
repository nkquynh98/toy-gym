from toy_gym.envs.toy_tasks.toy_pickplace2D import ToyPickPlace2D
from toy_gym.envs.toy_tasks.toy_pickplace_fiveobject import ToyPickPlaceFiveObject
from toy_gym.envs.tasks.pick_and_place_2d import PickAndPlace2D
from toy_gym.policy.ActionExecutor import ActionExecutor
from motion_planning.core.action import Action
import gym
import numpy as np
from numpngw import write_apng
import matplotlib.pyplot as plt
from toy_gym.policy.crude_policy import crude_policy
import time
import pybullet as p

from motion_planning.core.action import Action

from pyrieef.motion.trajectory import Trajectory, linear_interpolation_trajectory

def R_z(theta):
    cos_theta = np.cos(theta)
    sin_theta = np.sin(theta)
    return np.array([[cos_theta,-sin_theta],[sin_theta, cos_theta]])

def calculate_angle(vector):
    angle = np.arctan2(vector[1], vector[0])
    return angle
NUM_EPS = 5
MAX_STEPS = 10000
current_step = 0
# images = []
# #env = gym.make("ToyPickPlaceOneObject-v1",render=True)
# #env = ToyPickPlaceOneObject(render=True)
env = ToyPickPlaceFiveObject(render=True, map_name="maze_world")
obs = env.reset()
env.task.get_fixed_obstacles()
env.task.get_movable_obstacles()
# policy = crude_policy(env, False)
# done = False
q_init=env.robot.get_obs()[0:2]
q_final=env.task.get_obs()[0:2]
yaw = calculate_angle(q_final-q_init)
R_mat = R_z(yaw)
print(q_final)
print(yaw)
print(q_final-q_init)
q_final = q_final - R_mat@np.array([0.4,0.0])
print(q_final)


traj = linear_interpolation_trajectory(q_init=q_init,q_goal=q_final, T=20)
pick = Action("MoveToPick",traj=traj)
q_init=env.task.get_obs()[0:2]
q_final=env.task.get_goal()[0:2]
yaw = calculate_angle(q_final-q_init)
R_mat = R_z(yaw)
q_final = q_final - R_mat@np.array([0.4,0.0])
traj = linear_interpolation_trajectory(q_init=q_init,q_goal=q_final, T=20)
place = Action("MoveToPlace", traj=traj)
policy = ActionExecutor(env)
policy.set_action_list([pick, place])
done = False
#im = env.render("rgb_array",width=480, height=480,target_position=[0,0,0], yaw=-90, pitch=-90.1, distance=9.0)
#plt.imshow(im)
#plt.show()

for i in range(NUM_EPS):
    for _ in range(MAX_STEPS):
        action = policy.get_action()
        obs, reward, done, info = env.step(action)
        if done:
            break
    env.reset()
    policy.reset()
    #print("Reward", reward)

print("stop")