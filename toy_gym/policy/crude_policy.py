import numpy as np
import random
from toy_gym.envs.EnvTemplate import RobotTaskEnv

class crude_policy:
    def __init__(self, env: RobotTaskEnv, verbose=False, Kp=1, threshold = 0.01, threshold_angle = 0.1) -> None:
        self.TEMP_TARGET_REACHED = False
        self.IS_GRIP = False
        self.env = env
        self.robot = self.env.robot
        self.task = self.env.task
        self.current_obj_num = None
        self.verbose=verbose
        self.Kp = Kp
        self.threshold = threshold
        self.threshold_angle = threshold_angle
        self.get_object()
        pass
    def calculate_angle(self, vector):
        return np.arctan2(vector[1], vector[0])

    def get_object(self):
        undone_task = np.where(self.task.is_done() == False)[0]
        self.current_obj_num=random.choice(undone_task)
        
    def pick_and_place(self):
        action = np.array([0.,0.,0.])
        robot_obs = self.robot.get_obs()
        object_num = self.current_obj_num
        object_obs = self.task.get_obs()[0+object_num*2:2+object_num*2]
        target_obs = self.task.get_goal()[0+object_num*2:2+object_num*2]
        #action[1]=1
        if not self.TEMP_TARGET_REACHED:
            if not robot_obs[3]:
                linear_diff = object_obs - robot_obs[0:2]
                angle_diff = self.calculate_angle(linear_diff) - robot_obs[2]
                if self.verbose:
                    print("Going to object:", self.current_obj_num)
                    print("Robot angle: ", np.degrees(robot_obs[2]))
                    print("Angle to target : ", np.degrees(self.calculate_angle(linear_diff)))
                    print("linear_diff", np.linalg.norm(linear_diff))
                    print("angle_diff", angle_diff)
                if np.linalg.norm(linear_diff)>self.threshold:               
                    action[2] = angle_diff*self.Kp
                    if abs(angle_diff)<self.threshold_angle:
                        action[0:2]=linear_diff*self.Kp
                else:
                    if not self.IS_GRIP:
                        self.IS_GRIP = 1
            else:
                
                linear_diff = target_obs - robot_obs[0:2]
                angle_diff = self.calculate_angle(linear_diff) - robot_obs[2]
                if self.verbose:
                    print("Holding")
                    print("linear_diff", np.linalg.norm(linear_diff))
                    print("angle_diff", angle_diff)
                if np.linalg.norm(linear_diff)>self.threshold:               
                    action[2] = angle_diff*self.Kp
                    if abs(angle_diff)<self.threshold_angle:
                        action[0:2]=linear_diff*self.Kp
                else:
                    self.IS_GRIP = 0
                    self.TEMP_TARGET_REACHED = 1
        action_new = np.concatenate([action,[self.IS_GRIP]])
        return action_new

    def get_action(self):
        if self.TEMP_TARGET_REACHED:
           self.TEMP_TARGET_REACHED = False
           self.get_object()
        return self.pick_and_place() 