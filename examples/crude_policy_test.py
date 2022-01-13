from toy_gym.envs.toy_tasks.toy_pickplace2D import ToyPickPlace2D
from toy_gym.envs.toy_tasks.toy_pickplace_fiveobject import ToyPickPlaceFiveObject
from toy_gym.envs.tasks.pick_and_place_2d import PickAndPlace2D
import gym
from numpngw import write_apng
import matplotlib.pyplot as plt
from toy_gym.policy.crude_policy import crude_policy
import time
import pybullet as p

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

policy = crude_policy(env, True, is_constant_vel=False)
done = False
#im = env.render("rgb_array",width=480, height=480,target_position=[0,0,0], yaw=-90, pitch=-90.1, distance=9.0)
#plt.imshow(im)
#plt.show()
print("Sim", env.sim._bodies_idx)
print(env.get_workspace_objects().robot.position)
for i in range(NUM_EPS):
    for _ in range(MAX_STEPS):
        action = policy.get_action()
        obs, reward, done, info = env.step(action)
        print(done)
        if done:
            break
    env.reset()
    policy.reset()
    #print("Reward", reward)

print("stop")
