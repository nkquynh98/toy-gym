import gym
import matplotlib.pyplot as plt
from toy_gym.envs.toy_tasks.toy_pickplace2D import ToyPickPlace2D
import toy_gym
from stable_baselines3 import DDPG, HerReplayBuffer
#env = gym.make("ToyPickPlaceOneObject-v1",render=True)
env = ToyPickPlace2D("")
model = DDPG(policy="MultiInputPolicy", env=env, replay_buffer_class=HerReplayBuffer, verbose=1)

model.learn(total_timesteps=100000)
env.close()