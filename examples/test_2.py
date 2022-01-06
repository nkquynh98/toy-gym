from toy_gym.envs.robots.toy import Toy
from toy_gym.pybullet_gym import PyBullet
from toy_gym.envs.tasks.pick_and_place_2d import PickAndPlace2D
import numpy as np
import time
import matplotlib.pyplot as plt
sim = PyBullet(render=True,background_color=np.array([0,0,230]))
abc = Toy(sim)
object_positions = [[-1.0, -1.0], [3.0,3.0] , [3.0,2.0], [-2.0,3.0], [2.0,-4.0]]
#object_positions = [[-1.0, -1.3], [-1.0, -3] , [-1.0, -1.9], [-1.0, -2.2], [-1.0, -0.70]]
target_positions = [[2.0,4.0], [-2.5, 4.0], [-1.0, 4.0], [0.0, 4.0], [1.0, 4.0]]
task = PickAndPlace2D(sim, object_positions=object_positions, target_positions=target_positions)

#sim.loadURDF(body_name="Maze", fileName="panda_gym/envs/robots/urdf/maze_world.urdf")

#sim.place_visualizer(target_position=[0,0,0], yaw=90, pitch=-91, distance=10.0)
#Move to object 1
Kp=1
threshold = 0.01
threshold_angle = 0.1
IS_GRIP = 0
TARGET_REACHED = 0
OBJECT_NUM  = 0
print("action space: ",abc.action_space)
def calculate_angle(vector1, vector2):
    vector1 = vector1/np.linalg.norm(vector1)
    vector2 = vector2/np.linalg.norm(vector2)
    dot = np.dot(vector1, vector2)
    print("Dot", dot)
    return np.arccos(dot)
def calculate_angle(vector):
    return np.arctan2(vector[1], vector[0])
def pick_and_place(object_num, robot, task, IS_GRIP, TARGET_REACHED):
    action = np.array([0.,0.,0.])
    robot_obs = robot.get_obs()
    object_obs = task.get_obs()[0+object_num*2:2+object_num*2]
    target_obs = task.get_goal()[0+object_num*2:2+object_num*2]
    #action[1]=1
    if not TARGET_REACHED:
        if not robot_obs[3]:
            linear_diff = object_obs - robot_obs[0:2]
            angle_diff = calculate_angle(linear_diff) - robot_obs[2]
            print("Robot angle: ", np.degrees(robot_obs[2]))
            print("Angle to target : ", np.degrees(calculate_angle(linear_diff)))
            print("linear_diff", np.linalg.norm(linear_diff))
            print("angle_diff", angle_diff)
            if np.linalg.norm(linear_diff)>threshold:               
                action[2] = angle_diff*Kp
                if abs(angle_diff)<threshold_angle:
                    action[0:2]=linear_diff*Kp
            else:
                if not IS_GRIP:
                    IS_GRIP = 1
        else:
            print("Holding")
            linear_diff = target_obs - robot_obs[0:2]
            angle_diff = calculate_angle(linear_diff) - robot_obs[2]
            print("linear_diff", np.linalg.norm(linear_diff))
            print("angle_diff", angle_diff)
            if np.linalg.norm(linear_diff)>threshold:               
                action[2] = angle_diff*Kp
                if abs(angle_diff)<threshold_angle:
                    action[0:2]=linear_diff*Kp
            else:
                IS_GRIP = 0
                TARGET_REACHED = 1
    action_new = tuple((action,IS_GRIP))
    return action_new, IS_GRIP, TARGET_REACHED

reward = []
while(1):
    #action = abc.action_space.sample()
    # action = np.array([0.,0.,0.])
    # robot_obs = abc.get_obs()
    # object_obs = task.get_obs()[0:2]
    # target_obs = task.get_target()[0:2]
    # #action[1]=1
    # if not TARGET_REACHED:
    #     if not robot_obs[3]:
    #         linear_diff = object_obs - robot_obs[0:2]
    #         angle_diff = calculate_angle(linear_diff, np.array([1,0])) - robot_obs[2]
    #         print("linear_diff", np.linalg.norm(linear_diff))
    #         print("angle_diff", angle_diff)
    #         if np.linalg.norm(linear_diff)>threshold:               
    #             action[0:2]=linear_diff*Kp
    #             action[2] = angle_diff*Kp
    #         else:
    #             if not IS_GRIP:
    #                 IS_GRIP = 1
    #     else:
    #         print("Holding")
    #         linear_diff = target_obs - robot_obs[0:2]
    #         angle_diff = calculate_angle(linear_diff, np.array([1,0])) - robot_obs[2]
    #         print("linear_diff", np.linalg.norm(linear_diff))
    #         print("angle_diff", angle_diff)
    #         if np.linalg.norm(linear_diff)>threshold:               
    #             action[0:2]=linear_diff*Kp
    #             action[2] = angle_diff*Kp
    #         else:
    #             IS_GRIP = 0
    #             TARGET_REACHED = 1
    # action_new = tuple((action,IS_GRIP))
    #if OBJECT_NUM < len(object_positions):
    
    if not task.is_done():
        action_new, IS_GRIP, TARGET_REACHED= pick_and_place(OBJECT_NUM, abc, task, IS_GRIP, TARGET_REACHED)
        print("action_new",action_new)
        #print("Action ",action)

        abc.set_action(action_new)
        #print(abc.grasp_object())
        sim.step()
        if TARGET_REACHED:
            TARGET_REACHED = 0
            #task.ghost_object(OBJECT_NUM)
            OBJECT_NUM += 1
        print("Rewards", task.get_reward())    
        reward.append(task.get_reward())
        print("robot position", abc.get_obs())
        print("object position", task.get_obs())
        time.sleep(0.01)
    else:
        #plt.plot(reward)
        #plt.show()
        abc.reset()
        print("Finish")
    pass