import json
from toy_gym.envs.robots.toy_tamp import Toy_TAMP
from toy_gym.pybullet_gym import PyBullet
from toy_gym.envs.toy_tasks.toy_pickplace_tamp import ToyPickPlaceTAMP
import numpy as np

goal = {"object_0": "target_1", "object_1": "target_2", "object_2": "target_0", "object_3": "target_4", "object_4": "target_3"}

with open('/home/nkquynh/gil_ws/toy-gym/examples/sample_workspace.json') as f:
    data = json.load(f)
env = ToyPickPlaceTAMP(render=True, json_data = data, goal= goal)
while(1):
    print(env.get_geometric_state())
    print(env.get_logic_state())
    #task.reset()
    pass
print(data)