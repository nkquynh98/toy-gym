import re
from typing import Any, Dict, Union
from itertools import product
import numpy as np

from toy_gym.envs.EnvTemplate import Task
from toy_gym.pybullet_gym import PyBullet
from toy_gym.utils import distance
from toy_gym.envs.core.workspace_objects import Obstacle, Targets
import os
WORKSPACE_DIR = os.path.dirname(os.path.realpath(__file__))+"/../../../"

class PickAndPlace2D_TAMP(Task):
    def __init__(self, sim: PyBullet, json_data = None, goal = None, goal_threshold = 0.2, ghost_mode = True, verbose = False) -> None:
        super().__init__(sim)
        self.object_ids = {}
        self.target_ids = {}
        self.object_size = [0.1, 0.4]
        self.target_box_size = 0.2
        self.color_list = [[235.0,0.0,0.0],[77.0,48.0,194.0], [255.0,98.0,0.0], [0.0,205.0,0.0], [255.0,98.0,215.0]]
        self.object_color = {}
        self.goal_threshold = goal_threshold
        self.fixed_obstacles = {}
        self.movable_obstacles = {}
        self.targets_ws = {}
        #self.ghosted_object = {}
        self.ghost_mode = ghost_mode #Ghost the object after the object has reached the target
        self.verbose = verbose
        
        if json_data is not None:
            self.map_name=json_data["_map"]["_map_name"]
            self.object_list = json_data["_movable_obstacles"]
            self.target_list = json_data["_targets"]
            self.ghosted_object = np.zeros((len(self.object_list),),dtype=bool)
            if goal is not None:
                self.goal = goal
            else:
                self.goal = {}
                for object, target in zip(self.object_list, self.target_list):
                    self.goal[object]=target
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
        if self.object_list is not None:
            for i, object in enumerate(self.object_list.values()):
                object_name = object["_name"]
                self.object_color[object_name] = self.color_list[i%len(self.color_list)]
                self.object_ids[object_name]=self.sim.create_cylinder(
                    body_name=object_name,
                    radius=object["_dimension"][1],
                    height=object["_dimension"][0],
                    mass=0.01,
                    position=np.concatenate((object["_position"],[0.2])),
                    rgba_color=np.concatenate((np.array(self.object_color[object_name])/255,[1.0])),
                )
                target = self.target_list[self.goal[object_name]]
                targets_name = target["_name"]
                self.target_ids[targets_name]=self.sim.create_box(
                    body_name=targets_name,
                    half_extents=np.ones(3) * self.target_box_size / 2,
                    mass=0.0,
                    ghost=True,
                    position=np.concatenate((target["_position"],[0.0])),
                    rgba_color=np.concatenate((np.array(self.object_color[object_name])/255,[0.5])),
                )            
        else:
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
        number_of_object = len(self.object_list)
        for i in np.arange(number_of_object):
            distance_array=np.append(distance_array, np.linalg.norm(distance[2*i:2*i+2]))
        return distance_array
    def get_reward(self):
        return self.compute_reward(achieved_goal=self.get_achieved_goal(), desired_goal=self.get_goal())
    def get_achieved_goal(self) -> np.ndarray:
        observation = np.array([])
        for object in self.object_list:
            observation=np.append(observation, self.sim.get_base_position(object)[0:2])
        return observation
    def get_obs(self) -> np.ndarray:
        "Return the observation of the current position of the objects [object1_x, object1_y, object2_x, ...]"
        observation = np.array([])
        "Get the observation of the object"
        for object in self.object_list:
            observation=np.append(observation, self.sim.get_base_position(object)[0:2])
        return observation
    def get_goal(self):
        "Return the position of the objects target [target_for_object0_x, target_for_object0_y, target_for_object1_x, ...]"
        target_observation = np.array([])
        "Get the observation of the object"
        for object in self.object_list:
            target = self.goal[object]
            target_observation=np.append(target_observation, self.sim.get_base_position(target)[0:2])
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

    def get_logic_state(self):
        pred = []
        for target in self.target_ids:
            pred_child = ["target-free", target]
            for object in self.object_ids:
                object_pos = self.sim.get_base_position(object)[0:2]
                target_pos = self.sim.get_base_position(target)[0:2]
                if np.linalg.norm(object_pos - target_pos)<self.goal_threshold:
                    pred_child = ["at"]
                    pred_child.append(target)
                    pred_child.append(object)
            pred.append(pred_child)
        return pred
        

    
    def ghost_object(self, object_name):
        self.sim.delete_object(object_name)
        target = self.target_list[self.goal[object_name]]
        self.object_ids[object_name] = self.sim.create_cylinder(
                body_name=object_name,
                radius=self.object_list[object_name]["_dimension"][1],
                height=self.object_list[object_name]["_dimension"][0],
                mass=0.0,
                ghost=True,
                position=np.concatenate((target["_position"],[0.2])),
                rgba_color=np.concatenate((np.array(self.object_color[object_name])/255,[1.0])),
            )
    def is_target_reached(self, achieved_goal: np.ndarray, desired_goal: np.ndarray):
        distance_array = self.compute_distance(achieved_goal,desired_goal)
        is_reached = np.array(distance_array<self.goal_threshold)
        if self.ghost_mode:
            for i, object in enumerate(self.object_list):
                if is_reached[i] and not self.ghosted_object[i]:
                    self.ghost_object(object)
                    self.ghosted_object[i]=True
        return is_reached
    def is_success(self, achieved_goal: np.ndarray, desired_goal: np.ndarray, info: Dict[str, Any] = ...) -> Union[np.ndarray, float]:
        return self.is_target_reached(achieved_goal=achieved_goal,desired_goal=desired_goal).all()
    def is_done(self):
        return self.is_target_reached(self.get_achieved_goal(),self.get_goal())
    def reset(self, new_object_positions=None, new_target_positions=None) -> None:
        self.ghosted_object = np.zeros((len(self.object_list),),dtype=bool)
        for i, object in enumerate(self.object_list.values()):
            object_name = object["_name"]
            self.sim.delete_object(object_name)
            self.object_color[object_name] = self.color_list[i%len(self.color_list)]
            self.object_ids[object_name]=self.sim.create_cylinder(
                body_name=object_name,
                radius=object["_dimension"][1],
                height=object["_dimension"][0],
                mass=0.01,
                position=np.concatenate((object["_position"],[0.2])),
                rgba_color=np.concatenate((np.array(self.object_color[object_name])/255,[1.0])),
            )
            target = self.target_list[self.goal[object_name]]
            targets_name = target["_name"]
            self.target_ids[targets_name]=self.sim.create_box(
                body_name=targets_name,
                half_extents=np.ones(3) * self.target_box_size / 2,
                mass=0.0,
                ghost=True,
                position=np.concatenate((target["_position"],[0.0])),
                rgba_color=np.concatenate((np.array(self.object_color[object_name])/255,[0.5])),
            )    
        return None

