import re
from typing import Any, Dict, Union

import numpy as np

from toy_gym.envs.EnvTemplate import Task
from toy_gym.pybullet_gym import PyBullet
from toy_gym.utils import distance
from toy_gym.envs.core.workspace_objects import Obstacle, Targets
import os
WORKSPACE_DIR = os.path.dirname(os.path.realpath(__file__))+"/../../../"

class PickAndPlace2D(Task):
    def __init__(self, sim: PyBullet, object_positions=[[-1.0, -1.0]], target_positions=[[2.0,4.0]],map_name="maze_world",
                goal_threshold = 0.2,random_target=False, random_object=False,
                ghost_mode = True, verbose = False) -> None:
        super().__init__(sim)
        self.number_of_objects = len(object_positions)
        self.number_of_targets = len(target_positions)
        assert(self.number_of_objects == self.number_of_targets) #each target for each object
        self.object_positions = object_positions.copy()
        self.target_positions = target_positions.copy()
        self.object_ids = {}
        self.target_ids = {}
        self.object_size = [0.1, 0.4]
        self.target_box_size = 0.2
        self.color_list = [[235.0,0.0,0.0],[77.0,48.0,194.0], [255.0,98.0,0.0], [0.0,205.0,0.0], [255.0,98.0,215.0]]
        self.map_name=map_name
        self.random_target=random_target
        self.random_object=random_object
        self.goal_threshold = goal_threshold
        self.fixed_obstacles = {}
        self.movable_obstacles = {}
        self.targets_ws = {}
        self.ghosted_object = np.zeros((self.number_of_objects,),dtype=bool)
        self.ghost_mode = ghost_mode #Ghost the object after the object has reached the target
        self.verbose = verbose

        with self.sim.no_rendering():
            self._create_scence()
            self.sim.place_visualizer(target_position=[0,0,0], yaw=180, pitch=-90.1, distance=10.0)
            self.update_fixed_obstacles()
            self.update_movable_obstacles()
            _, map_dim = self.sim.get_link_shape(self.map_name)
            self.map_size = map_dim[0:2]
    def _create_scence(self):
        if self.map_name is not None:
            self.sim.loadURDF(body_name=self.map_name, fileName=WORKSPACE_DIR+"toy_gym/envs/urdf/"+self.map_name+".urdf")
        else:
            self.sim.create_plane(z_offset=-0.4)

        for i in np.arange(self.number_of_objects):
            body_name="object_{}".format(i)
            self.object_ids[body_name]=self.sim.create_cylinder(
                body_name=body_name,
                radius=self.object_size[0],
                height=self.object_size[1],
                mass=0.01,
                position=np.concatenate((self.object_positions[i],[0.2])),
                rgba_color=np.concatenate((np.array(self.color_list[i%len(self.color_list)])/255,[1.0])),
            )
            targets_name = "target_{}".format(i)
            self.target_ids[targets_name]=self.sim.create_box(
                body_name=targets_name,
                half_extents=np.ones(3) * self.target_box_size / 2,
                mass=0.0,
                ghost=True,
                position=np.concatenate((self.target_positions[i],[0.0])),
                rgba_color=np.concatenate((np.array(self.color_list[i%len(self.color_list)])/255,[0.5])),
            )
    def compute_reward(self, achieved_goal: np.ndarray, desired_goal: np.ndarray, info: Dict[str, Any] = ...) -> Union[np.ndarray, float]:
        return -np.sum(self.compute_distance(achieved_goal=achieved_goal,desired_goal=desired_goal))
    def compute_distance(self, achieved_goal: np.ndarray, desired_goal: np.ndarray):
        distance_array = np.array([])
        distance = desired_goal-achieved_goal
        for i in np.arange(self.number_of_objects):
            distance_array=np.append(distance_array, np.linalg.norm(distance[2*i:2*i+2]))
        return distance_array
    def get_reward(self):
        return self.compute_reward(achieved_goal=self.get_achieved_goal(), desired_goal=self.get_goal())
    def get_achieved_goal(self) -> np.ndarray:
        observation = np.array([])
        for i in self.number_of_objects:
            observation=np.append(observation, self.sim.get_base_position("object_{}".format(i))[0:2])
        return observation
    def get_obs(self) -> np.ndarray:
        "Return the observation of the current position of the objects [object1_x, object1_y, object2_x, ...]"
        observation = np.array([])
        "Get the observation of the object"
        for i in np.arange(self.number_of_objects):
            observation=np.append(observation, self.sim.get_base_position("object_{}".format(i))[0:2])
        return observation
    def get_goal(self):
        "Return the position of the objects target [object1_x, object1_y, object2_x, ...]"
        target_observation = np.array([])
        "Get the observation of the object"
        for i in np.arange(self.number_of_objects):
            target_observation=np.append(target_observation, self.sim.get_base_position("target_{}".format(i))[0:2])
        return target_observation

    def update_fixed_obstacles(self):
        obstacle_list = self.sim.get_links_names_and_ids(self.map_name)
        for obs_name, obs_index in obstacle_list.items():
            if obs_index!=-1:
                pos = self.sim.get_link_position(self.map_name, obs_index)[0:2]
                shape, dim = self.sim.get_link_shape(self.map_name, obs_index)
                dim = np.array(dim[0:2])
                obstacle = Obstacle(obs_name, shape, np.array(dim), pos, is_fixed=True)
                self.fixed_obstacles[obs_name]=obstacle
                if self.verbose:
                    print(obs_index)
                    print(pos)
                    print(shape)
                    print(dim)
                    print(obs_name)
        return self.fixed_obstacles
    def get_fixed_obstacles(self):
        return self.fixed_obstacles
    def update_movable_obstacles(self):
        obstacle_list = self.object_ids
        for obs_name, obs_index in obstacle_list.items():
            if obs_index!=-1:
                pos = self.sim.get_link_position(obs_name, -1)[0:2]
                shape, dim = self.sim.get_link_shape(obs_name, -1)
                dim = np.array(dim[0:2])
                #print("Obstacle dim", dim)
                obstacle = Obstacle(obs_name, shape, np.array(dim), pos, is_fixed=False)
                self.movable_obstacles[obs_name]=obstacle
                if self.verbose:
                    print(obs_index)
                    print(pos)
                    print(shape)
                    print(dim)
                    print(obs_name)
        return self.movable_obstacles
    def get_movable_obstacles(self):
        return self.movable_obstacles

    def update_ws_target_list(self):
        target_list = self.target_ids
        for target_name, target_index in target_list.items():
            if target_index!=-1:
                pos = self.sim.get_link_position(target_name, -1)[0:2]
                shape, dim = self.sim.get_link_shape(target_name, -1)
                dim = np.array(dim)
                target = Targets(target_name, shape, np.array(dim), pos, goal_radius=self.goal_threshold)
                self.targets_ws[target_name]=target
        return self.targets_ws
    def get_ws_target_list(self):
        return self.targets_ws


    def ghost_object(self, object_number):
        self.sim.delete_object("object_{}".format(object_number))
        self.sim.create_cylinder(
                body_name="object_{}".format(object_number),
                radius=self.object_size[0],
                height=self.object_size[1],
                mass=0.0,
                ghost=True,
                position=np.concatenate((self.target_positions[object_number],[0.2])),
                rgba_color=np.concatenate((np.array(self.color_list[object_number%len(self.color_list)])/255,[1.0])),
            )
    def is_target_reached(self, achieved_goal: np.ndarray, desired_goal: np.ndarray):
        distance_array = self.compute_distance(achieved_goal,desired_goal)
        is_reached = np.array(distance_array<self.goal_threshold)
        if self.ghost_mode:
            for i in np.arange(is_reached.shape[0]):
                if is_reached[i] and not self.ghosted_object[i]:
                    self.ghost_object(i)
                    self.ghosted_object[i]=True
        return is_reached
    def is_success(self, achieved_goal: np.ndarray, desired_goal: np.ndarray, info: Dict[str, Any] = ...) -> Union[np.ndarray, float]:
        return self.is_target_reached(achieved_goal=achieved_goal,desired_goal=desired_goal).all()
    def is_done(self):
        return self.is_target_reached(self.get_achieved_goal(),self.get_goal())
    def reset(self, new_object_positions=None, new_target_positions=None) -> None:
        if new_object_positions is not None:
            self.object_positions=new_object_positions
        if new_target_positions is not None:
            self.target_positions=new_target_positions
        self.number_of_objects = len(self.object_positions)    
        self.number_of_targets = len(self.target_positions)
        assert(self.number_of_objects == self.number_of_targets) #each target for each object
        self.ghosted_object = np.zeros((self.number_of_objects,),dtype=bool)
        
        for i in np.arange(self.number_of_objects):
            self.sim.delete_object("object_{}".format(i))
            body_name = "object_{}".format(i)
            self.object_ids[body_name]=self.sim.create_cylinder(
                body_name=body_name,
                radius=self.object_size[0],
                height=self.object_size[1],
                mass=0.01,
                position=np.concatenate((self.object_positions[i],[0.2])),
                rgba_color=np.concatenate((np.array(self.color_list[i%len(self.color_list)])/255,[1.0])),
            )
            if new_target_positions is not None:
                self.sim.delete_object("target_{}".format(i))
                target_name = "target_{}".format(i)
                self.target_ids[target_name]=self.sim.create_box(
                    body_name=target_name,
                    half_extents=np.ones(3) * self.target_box_size / 2,
                    mass=0.0,
                    ghost=True,
                    position=np.concatenate((self.target_positions[i],[0.0])),
                    rgba_color=np.concatenate((np.array(self.color_list[i%len(self.color_list)])/255,[0.5])),
            )
        return None

