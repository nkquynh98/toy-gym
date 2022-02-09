import numpy as np

from toy_gym.envs.EnvTemplate import RobotTaskEnv
from toy_gym.envs.robots.toy import Toy
from toy_gym.envs.tasks.pick_and_place_2d import PickAndPlace2D
from toy_gym.pybullet_gym import PyBullet
from toy_gym.envs.core.workspace_objects import *
class ToyPickPlace2D(RobotTaskEnv):
    def __init__(self, render: bool = False, 
                    object_positions=[[-1.0, -1.0]],
                    target_positions=[[2.0,4.0]],
                    is_target_random: bool = False,
                    is_object_random: bool = False,
                    no_of_random_object: int=1,
                    map_name="maze_world", map_size=(10,10), verbose=False) -> None:
        self.object_positions = object_positions
        #object_positions = [[-1.0, -1.3], [-1.0, -3] , [-1.0, -1.9], [-1.0, -2.2], [-1.0, -0.70]]
        self.target_positions = target_positions
        self.is_target_random = is_target_random
        self.is_object_random = is_object_random
        self.no_of_random_object = no_of_random_object
        self.map_size= map_size
        self.verbose = verbose
        sim = PyBullet(render=render,background_color=np.array([0,0,230]))
        robot = Toy(sim, verbose=verbose)
        
        if is_target_random or is_object_random:
            self.sample_random_object()
            assert (len(self.object_positions) == len(self.target_positions) == no_of_random_object) #Number of object must be equal with number of target
        
        task = PickAndPlace2D(sim,object_positions=self.object_positions, target_positions=self.target_positions, map_name=map_name, verbose=verbose)
        super().__init__(robot, task)
        if self.verbose:
            print("Object Position: ", self.object_positions)
            print("Target_Position: ", self.target_positions)
        #print("map size", self.map_size)

        self.workspace_objects = None
        self.init_workspace_objects()
    def init_workspace_objects(self):
        current_map = Map(self.task.map_name, self.task.map_size)
        fixed_obstacles = self.task.update_fixed_obstacles().copy()
        movable_obstacles = self.task.update_movable_obstacles()
        targets = self.task.update_ws_target_list()
        robot = self.robot.update_robot_ws()
        self.workspace_objects = Workspace_objects(map = current_map, robot=robot, fixed_obstacles=fixed_obstacles, 
                                    targets=targets, movable_obstacles=movable_obstacles)

    def get_workspace_objects(self):
        movable_obstacles = self.task.update_movable_obstacles()
        targets = self.task.update_ws_target_list()
        robot = self.robot.update_robot_ws()
        self.workspace_objects.targets = targets
        self.workspace_objects.robot = robot
        self.workspace_objects.movable_obstacles = movable_obstacles
        return self.workspace_objects
    def sample_random_object(self):
        np.random.seed()
        if self.is_target_random:
            self.target_positions = []
            for i in np.arange(self.no_of_random_object):
                self.target_positions.append((np.random.rand(2)-0.5)*(np.array(self.map_size)-np.array([0.5, 0.5])))
        

        if self.is_object_random:
            self.object_positions = []
            for i in np.arange(self.no_of_random_object):
                self.object_positions.append((np.random.rand(2)-0.5)*(np.array(self.map_size)-np.array([0.5, 0.5])))

    def reset(self):
        if self.is_target_random or self.is_object_random:
            self.sample_random_object()        
        with self.sim.no_rendering():
            self.robot.reset()
            self.task.reset(new_object_positions=self.object_positions)
        return self._get_obs()

    
