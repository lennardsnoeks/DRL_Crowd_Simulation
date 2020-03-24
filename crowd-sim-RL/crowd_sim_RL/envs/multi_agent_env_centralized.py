import gym
import numpy as np
import math
import copy
from gym import spaces
from utils.steerbench_parser import SimulationState
from visualization.visualize_training import VisualizationLive
from threading import Thread


class MultiAgentCentralized(gym.Env):

    def __init__(self, env_config):
        self.first_time = True
        self.time_step = 0.1

        self.reward_goal = 4
        #self.reward_goal = 5
        self.reward_collision = 5
        #self.reward_collision = 10
        self.reward_goal_reached = 10

        self.sim_state: SimulationState
        self.visualizer: VisualizationLive

        self.MIN_LIN_VELO = -0.5
        self.MAX_LIN_VELO = 1.5
        self.MAX_ANG_VELO = math.radians(45)

        self.WORLD_BOUND = 0

        self.mode = env_config["mode"]
        self.load_params(env_config["sim_state"])

        if "train" in self.mode:
            self.max_step_count = env_config["timesteps_reset"]

        self.step_counts_same = [0] * len(self.sim_state.agents)
        self.step_counts_obs = [0] * len(self.sim_state.agents)
        self.boxes = []
        for i in range(0, len(self.sim_state.agents)):
            box = [0] * 4
            self.boxes.append(box)

        self.action_space = self.make_action_space()

    def _set_visualizer(self, visualizer: VisualizationLive):
        self.visualizer = visualizer

    def load_params(self, sim_state: SimulationState):
        self.orig_sim_state = copy.deepcopy(sim_state)

        self.sim_state = sim_state

        self._load_world()

    def _load_world(self):
        self.steering_agents = self.sim_state.agents
        self.obstacles = self.sim_state.obstacles
        self.bounds = self.sim_state.clipped_bounds

        x_diff = (self.bounds[1] - self.bounds[0]) * 1.2
        y_diff = (self.bounds[3] - self.bounds[2]) * 1.2
        self.WORLD_BOUND = max(x_diff, y_diff)

        self.observation_space = self.make_observation_space()

    def make_action_space(self):
        min = []
        max = []
        for _ in self.sim_state.agents:
            min.append(self.MIN_LIN_VELO)
            min.append(-self.MAX_ANG_VELO)
            max.append(self.MAX_LIN_VELO)
            max.append(self.MAX_ANG_VELO)

        action_space = spaces.Box(np.array(min), np.array(max))

        return action_space

    def make_observation_space(self):
        min_pos = []
        max_pos = []

        for _ in self.sim_state.agents:
            min_pos.append(-self.WORLD_BOUND)
            min_pos.append(-self.WORLD_BOUND)
            max_pos.append(self.WORLD_BOUND)
            max_pos.append(self.WORLD_BOUND)

        num_agents = len(self.sim_state.agents)
        observation_space = spaces.Tuple((
            spaces.Box(np.array(min_pos), np.array(max_pos)),
            spaces.Box(low=-self.WORLD_BOUND, high=self.WORLD_BOUND,
                       shape=(num_agents,
                              self.sim_state.laser_history_amount,
                              self.sim_state.laser_amount + 1)),
            spaces.Box(low=0, high=20, shape=(num_agents,
                                              self.sim_state.laser_history_amount,
                                              self.sim_state.laser_amount + 1))
        ))

        return observation_space

    def initial_visualization(self, visualization):
        visualization.run()

    def setup_visualization(self):
        zoom_factor = 10
        visualization = VisualizationLive(self.sim_state, zoom_factor)
        thread = Thread(target=self.initial_visualization, args=(visualization,))
        thread.start()
        self._set_visualizer(visualization)
        self.first_time = False

    def step(self, action):
        if self.first_time and "vis" in self.mode:
            self.setup_visualization()

        reward = 0
        done = False
        reset_necessary = False

        internal_states = []
        external_states_laser = []
        external_states_type = []

        copy_agents = copy.deepcopy(self.steering_agents)

        for agent in self.steering_agents:
            i = agent.id * 2
            linear_vel = action[i]
            angular_vel = action[i + 1]

            if "train" in self.mode:
                linear_vel *= self.time_step
                angular_vel *= self.time_step

            # convert single orientation value (degrees or radians) to 2d representation | setup rotation matrix
            orientation_2d = np.array([[math.cos(agent.orientation)], [math.sin(agent.orientation)]])
            rotation_matrix = np.array([[math.cos(angular_vel), -math.sin(angular_vel)],
                                        [math.sin(angular_vel), math.cos(angular_vel)]])

            # calculate new position and orientation
            new_pos = np.add(agent.pos, (linear_vel * orientation_2d))
            new_ori_2d = np.matmul(rotation_matrix, orientation_2d)

            # convert calculated orientation back to polar value
            new_ori = math.atan2(new_ori_2d[1, 0], new_ori_2d[0, 0])

            # reward for getting closer to goal, if there are multiple goals, take closest one in consideration
            diff = 0
            max_distance_to_goal = 0
            shortest_goal = None
            first = True
            for goal in agent.goals:
                previous_distance_to_goal = math.sqrt(
                    (agent.pos[0, 0] - goal[0, 0]) ** 2 + (agent.pos[1, 0] - goal[1, 0]) ** 2)
                new_distance_to_goal = math.sqrt((new_pos[0, 0] - goal[0, 0]) ** 2 + (new_pos[1, 0] - goal[1, 0]) ** 2)
                diff = previous_distance_to_goal - new_distance_to_goal
                if first:
                    max_distance_to_goal = new_distance_to_goal
                    first = False
                if new_distance_to_goal <= max_distance_to_goal:
                    shortest_goal = goal
                if new_distance_to_goal < self.sim_state.goal_tolerance:
                    done = True
                    reward += self.reward_goal_reached
            reward += self.reward_goal * diff

            # clip and assign new position and orientation to the agent
            previous_pos = agent.pos
            new_pos = self._clip_pos(new_pos)
            agent.pos = new_pos
            agent.orientation = new_ori

            # check for collisions and assign rewards
            reward_collision, collision_obstacle = self._detect_collisions(agent, copy_agents)
            reward += reward_collision

            # represent the internal state of the agent (observation)
            internal_states = self._get_internal_state(agent, shortest_goal, internal_states)

            external_states_laser, external_states_type = self._get_external_state(agent, external_states_laser,
                                                                                   external_states_type)

            if "train" in self.mode:
                # When training, do manual reset once if the agent is stuck in local optima
                if self.step_counts_same[agent.id] == 0:
                    agent_x = previous_pos[0, 0]
                    agent_y = previous_pos[1, 0]
                    self.boxes[agent.id] = [agent_x - 1, agent_x + 1, agent_y - 1, agent_y + 1]
                elif self.step_counts_same[agent.id] == self.max_step_count:
                    reset_necessary = True

                if self.in_local_optima(agent):
                    self.step_counts_same[agent.id] += 1
                else:
                    self.step_counts_same[agent.id] = 0

                # Also do reset if the agent is wandering around inside obstacles for long time (like walls of hallway)
                if collision_obstacle:
                    self.step_counts_obs[agent.id] += 1
                else:
                    self.step_counts_obs[agent.id] = 0

                if self.step_counts_obs[agent.id] == self.max_step_count:
                    reset_necessary = True

        if "vis" in self.mode:
            self.render()

        internal_state = np.array(internal_states)
        observation = [internal_state, external_states_laser, external_states_type]

        if reset_necessary:
            observation = self.reset()
            reward = 0
            done = False
            for i in range(0, len(self.steering_agents)):
                self.step_counts_same[i] = 0
                self.step_counts_obs[i] = 0

        return observation, reward, done, {}

    def in_local_optima(self, agent):
        agent_x = agent.pos[0, 0]
        agent_y = agent.pos[1, 0]

        x_min = self.boxes[agent.id][0]
        x_max = self.boxes[agent.id][1]
        y_min = self.boxes[agent.id][2]
        y_max = self.boxes[agent.id][3]

        return x_min <= agent_x <= x_max and y_min <= agent_y <= y_max

    def _clip_pos(self, pos):
        if pos[0, 0] < self.bounds[0]:
            pos[0, 0] = self.bounds[0]
        elif pos[0, 0] > self.bounds[1]:
            pos[0, 0] = self.bounds[1]

        if pos[1, 0] < self.bounds[2]:
            pos[1, 0] = self.bounds[2]
        elif pos[1, 0] > self.bounds[3]:
            pos[1, 0] = self.bounds[3]

        return pos

    @staticmethod
    def _get_internal_state(agent, goal, internal_states):
        rotation_matrix_new = np.array([[math.cos(agent.orientation), -math.sin(agent.orientation)],
                                        [math.sin(agent.orientation), math.cos(agent.orientation)]])
        relative_pos_agent_to_goal = np.subtract(goal, agent.pos)
        internal_state = np.matmul(np.linalg.inv(rotation_matrix_new), relative_pos_agent_to_goal)

        internal_states.append(internal_state[0, 0])
        internal_states.append(internal_state[1, 0])

        return internal_states

    def _get_external_state(self, agent, external_states_laser, external_states_type):
        laser_distances = []
        agent.laser_lines = []
        types = []

        start_point = agent.orientation - math.radians(90)
        increment = math.radians(180 / self.sim_state.laser_amount)
        for i in range(0, self.sim_state.laser_amount + 1):
            laser_ori = start_point + i * increment
            x_ori = math.cos(laser_ori)
            y_ori = math.sin(laser_ori)
            distance, x_end, y_end, type = self._get_first_crossed_object(agent, x_ori, y_ori)
            laser_distances.append(distance)
            types.append(type)
            agent.laser_lines.append(np.array([x_end, y_end]))

        if len(agent.laser_history) == self.sim_state.laser_history_amount:
            agent.laser_history.pop(0)
            agent.type_history.pop(0)
        else:
            while len(agent.laser_history) < self.sim_state.laser_history_amount - 1:
                agent.laser_history.append(np.zeros(self.sim_state.laser_amount + 1))
                agent.type_history.append(np.zeros(self.sim_state.laser_amount + 1))
        agent.laser_history.append(np.array(laser_distances))
        agent.type_history.append(np.array(types))

        external_states_laser.append(agent.laser_history)
        external_states_type.append(agent.type_history)

        return external_states_laser, external_states_type

    def _get_first_crossed_object(self, current_agent, x_ori, y_ori):
        distance = 1000000
        x_end = 0
        y_end = 0
        type = 0
        iteration_step = 0.2
        collision = False

        x_agent = current_agent.pos[0, 0]
        y_agent = current_agent.pos[1, 0]

        distant_x = x_agent
        distant_y = y_agent
        distance_to_object = math.sqrt((x_agent - distant_x) ** 2 + (y_agent - distant_y) ** 2)

        #while distance_to_object < 10:
        while True:
            distant_x += x_ori * iteration_step
            distant_y += y_ori * iteration_step
            distance_to_object = math.sqrt((x_agent - distant_x) ** 2 + (y_agent - distant_y) ** 2)

            for agent in self.steering_agents:
                if current_agent.id == agent.id:
                    """for goal in agent.goals:
                        if self._point_in_circle(distant_x, distant_y, goal[0, 0], goal[1, 0],
                                                 self.sim_state.goal_tolerance):
                            if distance_to_object < distance:
                                distance = distance_to_object
                                x_end = distant_x
                                y_end = distant_y
                                collision = True
                                type = 0"""
                else:
                    if self._point_in_circle(distant_x, distant_y, agent.pos[0, 0], agent.pos[1, 0], agent.radius):
                        distance = distance_to_object
                        x_end = distant_x
                        y_end = distant_y
                        collision = True
                        type = 10

            for obstacle in self.obstacles:
                if obstacle.contains(distant_x, distant_y):
                    if distance_to_object < distance:
                        distance = distance_to_object
                        x_end = distant_x
                        y_end = distant_y
                        collision = True
                        type = 20
                        break

            if collision:
                break
            else:
                if self._collision_bound(distant_x, distant_y,
                                         self.bounds[0], self.bounds[1], self.bounds[2], self.bounds[3]):
                    if distance_to_object < distance:
                        distance = distance_to_object
                        x_end = distant_x
                        y_end = distant_y
                        type = 20
                        collision = True
                    break

        """if distance_to_object >= 10 and not collision:
            distance = 0
            x_end = distant_x
            y_end = distant_y"""

        return distance, x_end, y_end, type

    def _detect_collisions(self, current_agent, agents):
        reward = 0
        collision = False
        collision_obstacle = False

        # detect collision with obstacles
        for obstacle in self.obstacles:
            if self._collision_circle_rectangle(obstacle.x, obstacle.y, obstacle.width, obstacle.height,
                                                current_agent.pos[0, 0], current_agent.pos[1, 0], current_agent.radius):
                reward -= self.reward_collision
                collision = True
                collision_obstacle = True

        # detect collision with other agents
        if not collision:
            for agent in agents:
                if current_agent.id != agent.id:
                    if self._collision_circle_circle(current_agent.pos[0, 0], current_agent.pos[1, 0], current_agent.radius,
                                                     agent.pos[0, 0], agent.pos[1, 0], agent.radius):
                        reward -= self.reward_collision
                        collision = True

        # detect collision with world bounds
        if not collision:
            if self._collision_bound(current_agent.pos[0, 0], current_agent.pos[1, 0],
                                     self.bounds[0], self.bounds[1], self.bounds[2], self.bounds[3]):
                reward -= self.reward_collision

        return reward, collision_obstacle

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

    @staticmethod
    def _point_in_circle(x_p, y_p, x_c, y_c, radius):
        return (x_p - x_c) ** 2 + (y_p - y_c) ** 2 < radius ** 2

    @staticmethod
    def _collision_bound(x_p, y_p, x_min, x_max, y_min, y_max):
        return x_p <= x_min or x_p >= x_max or y_p <= y_min or y_p >= y_max

    def reset(self):
        self.sim_state = copy.deepcopy(self.orig_sim_state)

        self._load_world()

        internal_states = []
        external_states_laser = []
        external_states_type = []

        for agent in self.steering_agents:
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

            internal_states = self._get_internal_state(agent, shortest_goal, internal_states)
            external_states_laser, external_states_type = self._get_external_state(agent, external_states_laser,
                                                                                   external_states_type)

        observation = [internal_states, external_states_laser, external_states_type]

        return observation

    def get_agents(self):
        return self.steering_agents

    def render(self, mode='human'):
        self.visualizer.update_agents(self.steering_agents)

    def get_observation_space(self):
        return self.observation_space

    def get_action_space(self):
        return self.action_space
