import numpy as np
from gym import spaces

from toy_gym.envs.EnvTemplate import PyBulletRobot
from toy_gym.pybullet_gym import PyBullet
from toy_gym.envs.core.workspace_objects import Robot
import os 
WORKSPACE_DIR = os.path.dirname(os.path.realpath(__file__))+"/../../../"

import pybullet as p

class Toy(PyBulletRobot):
    def __init__(self, 
        sim: PyBullet, 
        base_position: np.ndarray = np.array([0.0, 0.0, 0.0]),
        base_rotation: np.ndarray = np.array([0,0,0])):
        #action_space = spaces.Tuple((spaces.Box(-0.5,0.5,(3,),dtype=np.float32),spaces.Discrete(2,)))
        action_space = spaces.Box(np.array([-0.5,-0.5,-1.0, 0.0]),np.array([0.5,0.5,1.0, 1.0]),dtype=np.float32)
        #print("Action sampled: ",action_space.sample()[0])
        self.urdf_file = WORKSPACE_DIR+"toy_gym/envs/urdf/toy_robot_2.urdf"
        super().__init__(
            sim,   
            body_name="Toy",
            file_name=WORKSPACE_DIR+"toy_gym/envs/urdf/toy_robot_2.urdf",
            base_position=base_position,
            action_space=action_space,
            joint_indices=np.array([0, 1, 2, 3, 4, 5, 6]),
            joint_forces=np.array([1000, 1000, 1000, 10, 10, 10, 10]),
        )
        self.is_holding = False
        self.robot_joints = self.sim.get_joints_names_and_ids(self.body_name).copy()
        test = {'gripper_to_left_hand'}
        print(self.robot_joints)
        self.robot_links = self.sim.get_links_names_and_ids(self.body_name).copy()
        print(self.robot_links)
        self.hand_ids= [self.robot_joints[b'gripper_to_left_hand'], self.robot_joints[b'gripper_to_right_hand']]
        self.movement_ids = [self.robot_joints[b'x'], self.robot_joints[b'y'], self.robot_joints[b'theta']]
        self.init_base_position = base_position
        self.init_base_rotation = base_rotation
        self.grasped_object = []
        self.robot_ws = None       
        self.init_robot_ws()
    def set_action(self, action: np.ndarray):
        action_x_vel, action_y_vel, action_theta_vel = action[0:3].copy() #x_vel, y_vel, theta_vel
        action_gripper = action[3] # 1 for holding command, 0 for releasing
        linVel = np.array([action_x_vel, action_y_vel, 0])
        angVel = np.array([0, 0, action_theta_vel])
        #Send the control command to the robot
        self.send_velocity_control(linearVel=linVel,angularVel=angVel)
        self.control_gripper(action_gripper)
        #self.sim.set_joint_angles(self.body_name,self.movement_ids, [0.0,0.0,0.0])
        #print(action_gripper)
        return None

    def init_robot_ws(self):
        robot_position = self.sim.get_base_position(self.body_name)[0:2]
        robot_yaw = self.sim.get_base_rotation(self.body_name, "euler")[2]
        robot_shape, robot_dim = self.sim.get_link_shape(self.body_name, link_index=self.robot_links['body']-4) #-4 because 4 links do not have visual shape
        robot_dim = np.array(robot_dim[0:2])
        gripper_pose = self.sim.get_joint_info_position(self.body_name, self.robot_joints[b'base_to_gripper'])
        gripper_pose =np.array(gripper_pose[0:2])
        self.robot_ws = Robot(robot_name=self.body_name, robot_shape=robot_shape, robot_dim=robot_dim+[0., 0.05], position=robot_position, yaw=robot_yaw,
             gripper_pose=gripper_pose, urdf_file=self.urdf_file)        
    
    
    def update_robot_ws(self):
        robot_position = self.sim.get_base_position(self.body_name)[0:2]
        robot_yaw = self.sim.get_base_rotation(self.body_name, "euler")[2]
        self.robot_ws.position = robot_position
        self.robot_ws.yaw = robot_yaw
        return self.robot_ws
    def send_velocity_control(self, linearVel, angularVel):
        "Send Velocity to control the robot: Linear Velocity (x,y) and yaw angular velocity (theta)"
        self.sim.set_base_velocity(self.body_name, linearVel, angularVel)
    def control_gripper(self, action_gripper, force=10, grasp_pos_left = -0.175):      
        if action_gripper>0.5:
            if not self.is_holding:
                self.is_holding=True
                #Close Gripper
                self.sim.control_joints(self.body_name, self.hand_ids, [grasp_pos_left, -grasp_pos_left], [force, force])
                self.grasp_object()
        else:
            if self.is_holding:
                self.is_holding = False
                self.sim.control_joints(self.body_name, self.hand_ids, [0, 0],[force, force])
                self.release_object()
        return None

    def grasp_object(self):
        env_item_list = self.sim.get_object_list()
        current_object_list = {}
        for key, value in env_item_list.items():
            if "object" in key:
                current_object_list[key] = value
        #Check the position of the object w.r.t the robot position
        if current_object_list is not None:
            for object in current_object_list:
                distance = (self.sim.get_link_position(self.body_name, self.robot_links["gripper"])-self.sim.get_base_position(object))[0:2]
                distance = np.linalg.norm(distance)
                if distance<0.2:
                    self.grasped_object.append(object)
                    constraint_name=self.body_name + "_hold_"+object
                    self.sim.set_kinematic_constraint(constraint_name, 
                                                self.body_name, self.robot_links["gripper"], object, -1, jointType=p.JOINT_FIXED)
                    #print("{} is grapsed".format(object))
            #print("No object is grapsed")

        else:
            print("No object on the ground")
        return current_object_list
    
    def release_object(self):
        while self.grasped_object: #loop over all holding object
            print("Grasped object", self.grasped_object)
            object = self.grasped_object.pop()
            constraint_name=self.body_name + "_hold_"+object
            self.sim.delete_kinematic_constraint(constraint_name)

    def get_obs(self) -> np.ndarray:
        "Observation of the robot in the environment: Position (x,y) + yaw + gripping"
        robot_position = self.sim.get_base_position(self.body_name)[0:2]
        robot_yaw = self.sim.get_base_rotation(self.body_name, "euler")[2]
        gripper_position = self.sim.get_link_position(self.body_name, self.robot_links['gripper'])[0:2]
        obs = np.concatenate((robot_position,[robot_yaw],[self.is_holding],gripper_position))
        return obs

    def reset(self) -> np.ndarray:
        self.sim.set_base_velocity(self.body_name, [0,0,0],[0,0,0])
        self.sim.set_joint_angles(self.body_name,self.movement_ids, [0.0,0.0,0.0])
        self.sim.set_base_pose(self.body_name, self.init_base_position, self.init_base_rotation)
        self.control_gripper(0)
        self.grasped_object = []
        self.robot_ws = None       
        self.init_robot_ws()
        return None

    
