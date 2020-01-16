import gym
import numpy as np
import math
import copy
import socket
import pickle
from gym import spaces

from utils.steerbench_parser import SimulationState
from visualization.visualize_steerbench import Visualization


class SingleAgentEnv(gym.Env):

    def __init__(self):
        self.time_step = 0.1

        self.accumulated_reward = 0
        self.reward_goal = 4
        self.reward_collision = 3
        self.reward_smooth1 = 4
        self.reward_smooth2 = 1

        self.sim_state: SimulationState = SimulationState()
        self.visualizer: Visualization

        self.MIN_LIN_VELO = -0.5
        self.MAX_LIN_VELO = 1.5
        self.MAX_ANG_VELO = math.radians(45)

        self.goal_tolerance = 2
        self.laser_amount = 10

        self.WORLD_BOUND = 10000

        self.action_space = spaces.Box(np.array([self.MIN_LIN_VELO, -self.MAX_ANG_VELO]),
                                       np.array([self.MAX_LIN_VELO, self.MAX_ANG_VELO]))
        self.observation_space = spaces.Box(np.array([-self.WORLD_BOUND, -self.WORLD_BOUND]),
                                            np.array([self.WORLD_BOUND, self.WORLD_BOUND]))

        #self.soc = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

    def load_params(self, sim_state):
        self.sim_state = sim_state
        self._load_world()

    def _load_world(self):
        self.steering_agents = []
        for agent in self.sim_state.agents:
            agent_copy = copy.copy(agent)
            self.steering_agents.append(agent_copy)
        self.obstacles = self.sim_state.obstacles
        self.bounds = self.sim_state.clipped_bounds

    def step(self, action):
        # action is [v,w] with v the linear velocity and w the angular velocity
        done = False
        reward = 0

        linear_vel = np.clip(action, self.MIN_LIN_VELO, self.MAX_LIN_VELO)[0]
        angular_vel = np.clip(action, -self.MAX_ANG_VELO, self.MAX_ANG_VELO)[1]
        agent = self.steering_agents[0]

        linear_vel_timestep = linear_vel * self.time_step
        angular_vel_timestep = angular_vel * self.time_step

        # convert single orientation value (degrees or radians) to 2d representation | setup rotation matrix
        orientation_2d = np.array([[math.cos(agent.orientation)], [math.sin(agent.orientation)]])
        rotation_matrix = np.array([[math.cos(angular_vel_timestep), -math.sin(angular_vel_timestep)],
                                    [math.sin(angular_vel_timestep), math.cos(angular_vel_timestep)]])

        # calculate new position and orientation
        new_pos = np.add(agent.pos, (linear_vel_timestep * orientation_2d))
        new_ori_2d = np.matmul(rotation_matrix, orientation_2d)

        # convert calculated orientation back to polar value
        new_ori = math.atan2(new_ori_2d[1, 0], new_ori_2d[0, 0])

        # reward for getting closer to goal, if there are multiple goals, take closest one in consideration
        diff = 0
        max_distance_to_goal = 0
        shortest_goal = None
        first = True
        for goal in agent.goals:
            previous_distance_to_goal = math.sqrt((agent.pos[0, 0] - goal[0, 0]) ** 2 + (agent.pos[1, 0] - goal[1, 0]) ** 2)
            new_distance_to_goal = math.sqrt((new_pos[0, 0] - goal[0, 0]) ** 2 + (new_pos[1, 0] - goal[1, 0]) ** 2)
            if first:
                max_distance_to_goal = new_distance_to_goal
                first = False
            if new_distance_to_goal <= max_distance_to_goal:
                shortest_goal = goal
                diff = previous_distance_to_goal - new_distance_to_goal
            if new_distance_to_goal < self.goal_tolerance:
                done = True
        reward += self.reward_goal * diff

        # smooth out reward
        reward += -self.reward_smooth1 * self._get_reward_smooth(linear_vel, self.MIN_LIN_VELO, self.MAX_LIN_VELO) \
                  -self.reward_smooth2 * self._get_reward_smooth(angular_vel, -self.MAX_ANG_VELO, self.MAX_ANG_VELO)

        # assign new position and orientation to the agent
        agent.pos = new_pos
        agent.orientation = new_ori

        # check for collisions and assign rewards
        reward += self._detect_collisions(agent)

        # represent the internal state of the agent (observation)
        internal_state = self._get_internal_state(agent.pos, agent.orientation, shortest_goal)
        observation = np.array([
            internal_state[0, 0],
            internal_state[1, 0]
        ])

        self.accumulated_reward += reward

        return observation, reward, done, {}

    def _get_internal_state(self, agent, goal):
        rotation_matrix_new = np.array([[math.cos(agent.orientation), -math.sin(agent.orientation)],
                                        [math.sin(agent.orientation), math.cos(agent.orientation)]])
        relative_pos_agent_to_goal = np.subtract(goal, agent.pos)
        internal_state = np.matmul(np.linalg.inv(rotation_matrix_new), relative_pos_agent_to_goal)

        return internal_state

    def _get_external_state(self, agent):
        laser_distances = []

        start_point = agent.orientation - math.radians(90)
        increment = math.radians(180 / self.laser_amount)
        for i in range(1, self.laser_amount + 1):
            laser_ori = start_point + i * increment
            x_ori = math.cos(laser_ori)
            y_ori = math.sin(laser_ori)
            distance = self._get_first_crossed_object(agent.pos[0, 0], agent.pos[1, 0], x_ori, y_ori)
            laser_distances.append(distance)

        agent.laser_history.append(laser_distances)
        if len(laser_distances) == self.laser_amount:
            agent.laser_history.pop(0)

    def _get_first_crossed_object(self, x_agent, y_agent, x_ori, y_ori):
        distance = 1000000
        iteration_step = 0.1
        while True:
            distant_x = x_agent +  x_ori * iteration_step
            distant_y = y_agent +  x_ori * iteration_step
            for agent in self.steering_agents:

                pass

            for object in self.obstacles:
                pass

            for bound in self.bounds:
                pass

            break

        return distance

    @staticmethod
    def _get_reward_smooth(x, x_min, x_max):
        return abs(min(x - x_min, 0)) + abs(max(x - x_max, 0))

    def _detect_collisions(self, current_agent):
        reward = 0
        # detect collision with obstacles
        for obstacle in self.obstacles:
            if self._collision_circle_rectangle(obstacle.x, obstacle.y, obstacle.width, obstacle.height,
                                                current_agent.pos[0, 0], current_agent.pos[1, 0], current_agent.radius):
                reward -= self.reward_collision

        # detect collision with other agents or if agent crosses world bounds
        for agent in self.steering_agents:
            if current_agent.id != agent.id:
                if self._collision_circle_circle(current_agent.pos[0, 0], current_agent.pos[1, 0], current_agent.radius,
                                                 agent.pos[0, 0], agent.pos[1, 0], current_agent.radius):
                    reward -= self.reward_collision
            if agent.pos[0, 0] < self.bounds[0] or agent.pos[0, 0] > self.bounds[1] or \
                    agent.pos[1, 0] < self.bounds[2] or agent.pos[1, 0] > self.bounds[3]:
                reward -= self.reward_collision

        return reward

    @staticmethod
    def _collision_circle_rectangle(x_rect, y_rect, width, height, x_circle, y_circle, r):
        x_test = x_circle
        y_test = y_circle

        if x_circle < x_rect:
            x_test = x_rect
        elif x_circle > x_rect + width:
            x_test = x_rect + width

        if y_circle < y_rect:
            y_test = y_rect
        elif y_circle > y_rect + height:
            y_test = y_rect + height

        distance = math.sqrt((x_circle - x_test) ** 2 + (y_circle - y_test) ** 2)

        if distance <= r:
            return True

        return False

    @staticmethod
    def _collision_circle_circle(x_c1, y_c1, r_c1, x_c2, y_c2, r_c2):
        value1 = abs(r_c1 - r_c2)
        value2 = math.sqrt((x_c1 - x_c2) ** 2 + (y_c1 - y_c2) ** 2)
        value3 = abs(r_c1 + r_c2)

        return value1 <= value2 <= value3

    def reset(self):
        self._load_world()

        agent = self.steering_agents[0]
        max_distance_to_goal = 0
        first = True
        shortest_goal = None
        for goal in agent.goals:
            distance_to_goal = math.sqrt((agent.pos[0, 0] - goal[0, 0]) ** 2 + (agent.pos[1, 0] - goal[1, 0]) ** 2)
            if first:
                max_distance_to_goal = distance_to_goal
                first = False
            if distance_to_goal <= max_distance_to_goal:
                shortest_goal = goal

        internal_state = self._get_internal_state(agent, shortest_goal)
        observation = np.array([
            internal_state[0, 0],
            internal_state[1, 0]
        ])

        return observation

    def render(self, mode='human'):
        self.visualizer.update_agents(self.steering_agents)
        #data_string = pickle.dumps(self.steering_agents)
        #self.soc.send(data_string)

    def set_visualizer(self, visualizer: Visualization):
        self.visualizer = visualizer
        #self.establish_connection()

    """def establish_connection(self):
        host = "127.0.0.1"
        port = 8000
        try:
            self.soc.connect((host, port))
            print("succesfully connected")
        except:
            print("Connection Error")"""
