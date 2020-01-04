import gym
import numpy as np
import math

from gym import spaces
from utils.steerbench_parser import XMLSimulationState, SimulationState


class SingleAgentEnv(gym.Env):

    def __init__(self, test_file):
        self.step_count = 0
        self.time_step = 0.1

        self.reward_goal = 4
        self.reward_collision = 3

        self.test_file = test_file
        self.xml_sim_state = XMLSimulationState()

        self.reset()

        self.action_space = spaces.Box(np.array([-0.5, -(math.radians(45))]), np.array([1.5, math.radians(45)]))
        self.observation_space = spaces.

    def _load_world(self):
        self.steering_agents, self.obstacles, self.bounds = self.xml_sim_state.parse_xml(self.test_file)

    def step(self, action):
        # action is [v,w] with v the linear velocity and w the angular velocity
        done = False
        reward = 0
        self.step_count += 1
        linear_vel = action[0]
        angular_vel = action[1]
        agent = self.steering_agents[0]

        # set agent pos in 2x1 matrix
        agent_pos = np.array([[agent.pos[0]],[agent.pos[1]]])

        # convert single orientation value (degrees or radians) to 2d representation | setup rotation matrix
        orientation_2d = np.array([[math.cos(agent.orientation)],[math.sin(agent.orientation)]])
        rotation_matrix = np.array([[math.cos(angular_vel), -math.sin(angular_vel)], [math.sin(angular_vel), math.cos(angular_vel)]])

        # calculate new position and orientation
        new_pos = agent_pos + linear_vel * self.step_count * self.time_step * orientation_2d
        new_ori_2d = np.matmul(rotation_matrix, orientation_2d)

        # reward for getting closer to goal, if there are multiple goals, take closest one in consideration
        max_distance_to_goal = 1000000
        diff = 0
        shortest_goal = None
        for goal in agent.goals:
            distance_to_goal = math.sqrt((agent.pos[0] - goal[0]) ** 2 + (agent.pos[1] - goal[1]) ** 2)
            new_distance_to_goal = math.sqrt((agent.pos[0] - goal[0]) ** 2 + (agent.pos[1] - goal[1]) ** 2)
            if distance_to_goal < max_distance_to_goal:
                diff = distance_to_goal - new_distance_to_goal
                shortest_goal = goal
            if distance_to_goal == 0:
                done = True
        reward += self.reward_goal * diff

        # convert calculated orientation back to polar value
        new_ori = math.atan(new_ori_2d[1] / new_ori_2d[0])

        rotation_matrix_new = np.array([[math.cos(new_ori), -math.sin(new_ori)], [math.sin(new_ori), math.cos(new_ori)]])
        relative_pos_agent_to_goal = np.subtract(shortest_goal, new_pos)
        internal_state = np.matmul(np.linalg.inv(rotation_matrix_new), relative_pos_agent_to_goal)

        # check for collisions and assign rewards
        reward += self._detect_collisions(agent)

        observation = [

        ]

        # assign new position and orientation to the agent
        agent.pos = new_pos
        agent.orientation = new_ori

        return observation, reward, done, {}

    def _detect_collisions(self, current_agent):
        reward = 0
        # detect collision with obstacles
        for obstacle in self.obstacles:
            if self._collision_circle_rectangle(obstacle.x, obstacle.y, obstacle.width, obstacle.height, current_agent.pos[0], current_agent.pos[1], current_agent.radius):
                reward -= self.reward_collision

        # detect collision with other agents or if agent crosses world bounds
        for agent in self.steering_agents:
            if current_agent.id != agent.id:
                if self._collision_circle_circle(current_agent.pos[0], current_agent.pos[1], current_agent.radius, agent.pos[0], agent.pos[1], current_agent.radius):
                    reward -= self.reward_collision
                if agent.pos[0] < self.bounds[0] or agent.pos[0] > self.bounds[1] or agent.pos[1] < self.bounds[2] or agent.pos[1] > self.bounds[3]:
                    reward -= self.reward_collision

        return reward

    def _collision_circle_rectangle(self, x_rect, y_rect, width, height, x_circle, y_circle, r):
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

        x_dist = x_circle - x_test
        y_dist = y_circle - y_test
        distance = math.sqrt((x_dist * x_dist) / (y_dist * y_dist))

        if distance <= r:
            return True

        return False

    def _collision_circle_circle(self, x_c1, y_c1, r_c1, x_c2, y_c2, r_c2):
        value1 = abs(r_c1 - r_c2)
        value2 = math.sqrt((x_c1 - x_c2)**2 + (y_c1 - y_c2)**2)
        value3 = abs(r_c1 + r_c2)

        return value1 <= value2 <= value3


    def reset(self):
        self.step_count = 0

        self._load_world()



    def render(self, mode='human'):
        pass

