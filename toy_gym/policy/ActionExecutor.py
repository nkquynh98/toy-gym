from os import error
from numpy.testing import KnownFailureException
from toy_gym.envs.toy_tasks.toy_pickplace2D import ToyPickPlace2D
from motion_planning.core.action import Action
from typing import List
import numpy as np
class ActionExecutor:
    def __init__(self, env: ToyPickPlace2D, action_list: List[Action]=[], Kp=1, threshold = 0.05, threshold_angle = 0.1, 
                    is_constant_vel=False, linear_vel= 1, angular_vel=0.5):
        self.action_list = action_list
        self.current_action = None
        self.env = env
        self.robot = self.env.robot
        self.task = self.env.task
        self.Kp = Kp
        self.threshold = threshold
        self.threshold_angle = threshold_angle
        self.is_constant_vel = is_constant_vel
        self.current_waypoint = None
        self.linear_vel = linear_vel
        self.angular_vel = angular_vel
        self.robot_obs = self.robot.get_obs()
        if self.action_list:
            self.current_waypoint_index = 0
            self.current_action_index = 0
            self.update_waypoint_and_action()


    def set_action_list(self, action_list: List[Action]):
        self.action_list = action_list
        self.current_waypoint_index = 0
        self.current_action_index = 0
        self.update_waypoint_and_action()

    def move_to_way_point(self, waypoint: np.ndarray):
        robot_obs= self.robot.get_obs()
        robot_pos = robot_obs[0:2]
        robot_yaw = np.clip(robot_obs[2], -3.07178, 3.07178)
        linear_diff = waypoint - robot_pos
        action = np.zeros(4)
        angle_diff = self.calculate_angle(waypoint-robot_pos) - robot_yaw
        if np.linalg.norm(linear_diff)>self.threshold:
            if abs(angle_diff)<self.threshold_angle:
                action[0:2]=np.sign(linear_diff)*self.angular_vel if self.is_constant_vel else linear_diff*self.Kp               
            action[2]= np.sign(angle_diff)*self.angular_vel if self.is_constant_vel else angle_diff*self.Kp
        return action
    def update_robot_obs(self):
        self.robot_obs = self.robot.get_obs()
    def calculate_angle(self, vector):
        angle = np.arctan2(vector[1], vector[0])
        return np.clip(angle, -3.07178, 3.07178)
    def check_reach_waypoint(self, waypoint):
        robot_pos = self.robot_obs[0:2]
        linear_diff = waypoint - robot_pos
        if np.linalg.norm(linear_diff)<self.threshold:
            return True
        else:
            return False
    def update_waypoint_and_action(self):
        self.current_action = self.action_list[self.current_action_index]
        self.current_waypoint = self.current_action.trajectory.configuration(self.current_waypoint_index)
        #print("Current waypoint", self.current_waypoint)
    def get_action(self):
        if self.action_list is None:
            raise RuntimeError("Action list is not set")
        self.update_robot_obs()
        action = np.zeros(4)
        if self.current_action.name=="MoveToPlace":
            action[3] = 1
        #Check if it has reached the waypoint
        if not self.check_reach_waypoint(self.current_waypoint):
            #If not 
            action[0:3] = self.move_to_way_point(self.current_waypoint)[0:3]
        else:
            if self.current_waypoint_index == self.current_action.trajectory._T:
                if self.current_action.name == "MoveToPick":
                    action[3]=1
                elif self.current_action.name == "MoveToPlace":
                    action[3]=0
                #Check the final action has executed                
                if self.current_action_index == len(self.action_list)-1:
                    pass
                else:
                    self.current_action_index += 1
                    self.current_waypoint_index = 0
                    self.update_waypoint_and_action()

            else:
                self.current_waypoint_index+=1
                self.update_waypoint_and_action()
        
        return action
    

        
