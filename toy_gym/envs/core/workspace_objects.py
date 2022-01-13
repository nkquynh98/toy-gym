import numpy as np
from typing import List
class Obstacle(object):
    def __init__(self, name: str, shape: str, dimension:np.ndarray, position, is_fixed):
        self._name = name
        self._shape = shape
        self._dimension = dimension
        self._position = position
        self._is_fixed = is_fixed


    @property
    def shape(self):
        return self._shape
    @property
    def name(self):
        return self._name
    @name.setter
    def name(self, new_name:str):
        self._name = new_name

    @property
    def position(self):
        return self._position
    @position.setter
    def position(self, position:np.ndarray):
        self._position = position

    @property
    def is_fixed(self):
        return self._is_fixed
    @is_fixed.setter
    def is_fixed(self, is_fixed: bool):
        self._is_fixed = is_fixed

    @property
    def dimension(self):
        return self._dimension
    @dimension.setter
    def name(self, dimension: np.ndarray):
        self._dimension = dimension    

class Robot(object):
    def __init__(self, robot_name, robot_shape: str, robot_dim: np.ndarray, position: np.ndarray, yaw: float, gripper_pose: np.ndarray = None, urdf_file: str = None
                ):
        self._robot_name = robot_name
        self._robot_shape = robot_shape
        self._robot_dim = robot_dim
        self._position = position
        self._yaw = yaw
        self._gripper_pose = gripper_pose
    
    @property
    def robot_name(self):
        return self._robot_name
    @robot_name.setter
    def robot_name(self, robot_name:str):
        self._robot_name = robot_name

    @property
    def robot_shape(self):
        return self._robot_shape
    @robot_shape.setter
    def robot_shape(self, robot_shape:str):
        self._robot_shape = robot_shape    

    @property
    def robot_dim(self):
        return self._robot_dim
    @robot_dim.setter
    def robot_dim(self, robot_dim: np.ndarray):
        self._robot_dim = robot_dim    

    @property
    def position(self):
        return self._position
    @position.setter
    def position(self, position:np.ndarray):
        self._position = position     
    @property
    def yaw(self):
        return self._yaw
    @yaw.setter
    def yaw(self, yaw: np.ndarray):
        self._yaw = yaw         
class Map(object):
    def __init__(self, map_name: str, map_dim: np.ndarray):
        self._map_name = map_name
        self._map_dim = map_dim
    @property
    def map_name(self):
        return self._map_name
    @map_name.setter
    def map_name(self, map_name: str):
        self._map_name = map_name    

    @property
    def map_dim(self):
        return self._map_dim
    @map_dim.setter
    def map_dim(self, map_dim: np.ndarray):
        self._map_dim = map_dim 

class Targets(Obstacle):
    def __init__(self, name: str, shape: str, dimension:np.ndarray, position, goal_radius: float):
        self._goal_radius = goal_radius
        super().__init__(name,shape,dimension,position,True)
    @property
    def goal_radius(self):
        return self._goal_radius
    @goal_radius.setter
    def goal_radius(self, goal_radius: float):
        self._goal_radius = goal_radius

class Workspace_objects(object):
    def __init__(self, map: Map, robot: Robot, fixed_obstacles: List[Obstacle], targets: List[Targets], movable_obstacles: List[Obstacle]):
        self._map = map
        self._robot = robot
        self._fixed_obstacles = fixed_obstacles.copy()
        self._movable_obstacles = movable_obstacles.copy()
        self._targets = targets.copy()
    @property
    def map(self):
        return self._map

    @property
    def fixed_obstacles(self):
        return self._fixed_obstacles
    @property
    def robot(self):
        return self._robot
    @robot.setter
    def robot(self, robot: Robot):
        self._robot = robot
    @property
    def movable_obstacles(self):
        return self._movable_obstacles
    @movable_obstacles.setter
    def movable_obstacles(self, movable_obstacles: List[Obstacle]):
        self._movable_obstacles = movable_obstacles.copy()

    @property
    def targets(self):
        return self._targets
    @targets.setter
    def targets(self, targets: List[Targets]):
        self._targets = targets.copy()

    

    