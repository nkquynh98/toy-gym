import numpy as np
import random
from toy_gym.envs.EnvTemplate import RobotTaskEnv

class crude_policy:
    def __init__(self, env: RobotTaskEnv, verbose=False, Kp=1, threshold = 0.1, threshold_angle = 0.1, 
                    is_constant_vel=False, linear_vel= 1, angular_vel=0.5) -> None:
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
        self.linear_vel = linear_vel
        self.angular_vel = angular_vel
        self.is_constant_vel = is_constant_vel
        self.get_object()
        self.old_angle_sign = 1
        pass
    def calculate_angle(self, vector):
        angle = np.arctan2(vector[1], vector[0])
        return np.clip(angle, -3.07178, 3.07178)

    def get_object(self):
        undone_task = np.where(self.task.is_done() == False)[0]
        if undone_task.size!=0:
            self.current_obj_num=random.choice(undone_task)
        else:
            self.current_obj_num=None
        
    def pick_and_place(self):
        action = np.array([0.,0.,0.])
        robot_obs = self.robot.get_obs()
        robot_pos = robot_obs[0:2]
        gripper_pos = robot_obs[4:6]
        object_num = self.current_obj_num
        object_obs = self.task.get_obs()[0+object_num*2:2+object_num*2]
        target_obs = self.task.get_goal()[0+object_num*2:2+object_num*2]
        #action[1]=1
        #Avoid fluctuation between +179 degree and -179 degree, threshold is 176 degree

        robot_obs[2] = np.clip(robot_obs[2], -3.07178, 3.07178)
        if not self.TEMP_TARGET_REACHED:
            if not robot_obs[3]:
                linear_diff = object_obs - gripper_pos
                angle_diff = self.calculate_angle(object_obs-robot_pos) - robot_obs[2]
                if self.verbose:
                    print("Going to object:", self.current_obj_num)
                    print("Robot angle: ", np.degrees(robot_obs[2]))
                    print("Angle to target : ", np.degrees(self.calculate_angle(linear_diff)))
                    print("linear_diff", np.linalg.norm(linear_diff))
                    print("angle_diff", angle_diff)
                if np.linalg.norm(linear_diff)>self.threshold:
                    if abs(angle_diff)<self.threshold_angle:
                        action[0:2]=np.sign(linear_diff)*self.angular_vel if self.is_constant_vel else linear_diff*self.Kp               
                    action[2]= np.sign(angle_diff)*self.angular_vel if self.is_constant_vel else angle_diff*self.Kp
                else:
                    if not self.IS_GRIP:
                        self.IS_GRIP = 1
            else:
                linear_diff = target_obs - gripper_pos
                angle_diff = self.calculate_angle(target_obs-robot_pos) - robot_obs[2]
                if self.verbose:
                    print("Holding object:", self.current_obj_num)
                    print("Robot angle: ", np.degrees(robot_obs[2]))
                    print("Angle to target : ", np.degrees(self.calculate_angle(linear_diff)))
                    print("linear_diff", np.linalg.norm(linear_diff))
                    print("angle_diff", angle_diff)
                if np.linalg.norm(linear_diff)>self.threshold:
                    if abs(angle_diff)<self.threshold_angle:
                        action[0:2]=np.sign(linear_diff)*self.angular_vel if self.is_constant_vel else linear_diff*self.Kp               
                    action[2]= np.sign(angle_diff)*self.angular_vel if self.is_constant_vel else angle_diff*self.Kp
                else:
                    self.IS_GRIP = 0
                    self.TEMP_TARGET_REACHED = 1
        action_new = np.concatenate([action,[self.IS_GRIP]])
        return action_new
    def stand_still(self):
        return np.array([0.,0.,0.,0.])
    def get_action(self):
        if self.TEMP_TARGET_REACHED:
           self.TEMP_TARGET_REACHED = False
           self.get_object()
        if self.current_obj_num is not None:
            return self.pick_and_place()
        else:
            return self.stand_still() 

    def reset(self):
        self.TEMP_TARGET_REACHED = False
        self.IS_GRIP = 0
        self.get_object()