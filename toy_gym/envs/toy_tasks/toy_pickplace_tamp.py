import numpy as np
from toy_gym.envs.EnvTemplate import RobotTaskEnv
from toy_gym.envs.tasks.pick_and_place_2d_tamp import PickAndPlace2D_TAMP
from toy_gym.envs.robots.toy_tamp import Toy_TAMP
from toy_gym.pybullet_gym import PyBullet
from toy_gym.envs.core.workspace_objects import *
class ToyPickPlaceTAMP(RobotTaskEnv):
    def __init__(self, render: bool = False, json_data = None, goal = None, verbose=False) -> None:
        #object_positions = [[-1.0, -1.3], [-1.0, -3] , [-1.0, -1.9], [-1.0, -2.2], [-1.0, -0.70]]
    
        self.map_size= json_data["_map"]["_map_dim"]
        self.verbose = verbose
        sim = PyBullet(render=render,background_color=np.array([0,0,230]))
        robot_init_pos = np.array([json_data["_robot"]["_position"][0], json_data["_robot"]["_position"][1], 0.0])
        robot_init_rot = np.array([0,0, json_data["_robot"]["_yaw"]])
        robot = Toy_TAMP(sim, base_position=robot_init_pos, base_rotation=robot_init_rot, verbose=verbose)
        task_goal = self.process_goal(goal)
        # task_goal = {"object_0": "target_1", "object_1": "target_2", "object_2": "target_0", "object_3": "target_4", "object_4": "target_3"}
        task = PickAndPlace2D_TAMP(sim, json_data = json_data, goal = task_goal, verbose=verbose)
        super().__init__(robot, task)
        if self.verbose:
            print("Object Position: ", self.object_positions)
            print("Target_Position: ", self.target_positions)
        #print("map size", self.map_size)

        self.workspace_objects = None
        self.init_workspace_objects()

    def process_goal(self, goal):
        task_goal = {}
        for sub_goal in goal[0]:
            if sub_goal[0] == "at":
                task_goal[sub_goal[1]]=sub_goal[2]
        return task_goal
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


    def get_geometric_state(self):
        return self.get_workspace_objects().get_dict()

    def get_logic_state(self):
        robot_logic_state = self.robot.get_logic_state()
        env_logic_state = self.task.get_logic_state()
        return [*robot_logic_state, *env_logic_state]
    def reset(self):   
        with self.sim.no_rendering():
            self.robot.reset()
            self.task.reset()
        return self._get_obs()
        
