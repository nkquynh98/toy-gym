from toy_gym.envs.toy_tasks.toy_pickplace_oneobject import ToyPickPlaceOneObject
from toy_gym.envs.toy_tasks.toy_pickplace_fiveobject import ToyPickPlaceFiveObject
import gym
from numpngw import write_apng
import matplotlib.pyplot as plt
from toy_gym.policy.crude_policy import crude_policy
images = []
#env = gym.make("ToyPickPlaceOneObject-v1",render=True)
#env = ToyPickPlaceOneObject(render=True)
env = ToyPickPlaceFiveObject(render=True, map_name="maze_world")
#env = ToyPickPlaceFiveObject(render=True)
obs = env.reset()
policy = crude_policy(env, False)
done = False
#im = env.render("rgb_array",width=480, height=480,target_position=[0,0,0], yaw=-90, pitch=-90.1, distance=9.0)
#plt.imshow(im)
#plt.show()
print("Sim", env.sim._bodies_idx)
while not done:
    action = policy.get_action()
    obs, reward, done, info = env.step(action)
    #print("Reward", reward)

print("stop")

#env.close()