import gym
import numpy as np
import math
import copy
from gym import spaces
from utils.steerbench_parser import SimulationState


class SingleAgentEnv3(gym.Env):

    def __init__(self, env_config):
        self.id = env_config["agent_id"]
        self.time_step = 0.1

        self.reward_goal = 5
        self.reward_collision = 15
        self.reward_collision_clip = 5
        #self.reward_goal_reached = 5
        self.reward_goal_reached = 5
        self.reset_pos_necessary = False

        self.sim_state: SimulationState

        self.MIN_LIN_VELO = -0.5
        self.MAX_LIN_VELO = 1.5
        self.MAX_ANG_VELO = math.radians(45)

        self.WORLD_BOUND = 0

        self.action_space = spaces.Box(np.array([self.MIN_LIN_VELO, -self.MAX_ANG_VELO]),
                                       np.array([self.MAX_LIN_VELO, self.MAX_ANG_VELO]))

        self.mode = env_config["mode"]
        self.load_params(env_config["sim_state"])

        self.step_count = 0
        self.step_count_same = 0
        self.step_count_obs = 0

        # keep IDs of agents current agent collided with, if they collide again in next timestep don't add
        # another negative reward, becomes too large and agents won't go near each other again
        self.collision_ids_agent = []
        self.collision_ids_obs = []

        if "train" in self.mode:
            self.max_step_count = env_config["timesteps_reset"]

    def load_params(self, sim_state: SimulationState):
        self.orig_sim_state = copy.deepcopy(sim_state)

        self.sim_state = sim_state

        self._load_world()

        x_diff = (self.bounds[1] - self.bounds[0]) * 1.2
        y_diff = (self.bounds[3] - self.bounds[2]) * 1.2
        self.WORLD_BOUND = math.ceil(max(x_diff, y_diff))

        self.observation_space = spaces.Tuple((
            spaces.Box(np.array([-self.WORLD_BOUND, -self.WORLD_BOUND]),
                       np.array([self.WORLD_BOUND, self.WORLD_BOUND])),
            spaces.Box(low=0, high=self.WORLD_BOUND * 2,
                       shape=(self.sim_state.laser_history_amount, self.sim_state.laser_amount + 1)),
            spaces.Box(low=0, high=2, shape=(self.sim_state.laser_history_amount, self.sim_state.laser_amount + 1))
        ))

    def _load_world(self):
        self.steering_agents = self.sim_state.agents
        if "multi" not in self.mode:
            self.compare_agents = self.steering_agents
        self.obstacles = self.sim_state.obstacles
        self.bounds = self.sim_state.clipped_bounds

    def step(self, action):
        reward = 0
        done = False

        linear_vel = action[0]
        angular_vel = action[1]
        agent = self.steering_agents[self.id]

        # set agent pos back to intial state, neceassary in some cases (see _check_resets method)
        if self.reset_pos_necessary:
            agent.pos[0, 0] = self.orig_sim_state.agents[agent.id].pos[0, 0]
            agent.pos[1, 0] = self.orig_sim_state.agents[agent.id].pos[1, 0]
            agent.orientation = self.orig_sim_state.agents[agent.id].orientation
            self.reset_pos_necessary = False
            #reward = 0

        # adjust linear and angular velocity according to timestep
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
            previous_distance_to_goal = self._calculate_distance_goal(agent.pos, goal)
            new_distance_to_goal = self._calculate_distance_goal(new_pos, goal)
            diff = previous_distance_to_goal - new_distance_to_goal
            if first:
                max_distance_to_goal = new_distance_to_goal
                first = False
            if new_distance_to_goal <= max_distance_to_goal:
                shortest_goal = goal
            if new_distance_to_goal <= self.sim_state.goal_tolerance:
                done = True
                agent.done = True
                self.step_count_same = 0
                self.step_count_obs = 0
                self.step_count = 0
                reward += self.reward_goal_reached
        reward += self.reward_goal * diff

        # assign new pos/ori
        previous_pos = agent.pos
        agent.pos = new_pos
        agent.orientation = new_ori

        # check for collisions and assign rewards
        reward_collision, collision_obstacle, collided_obs_id = self._detect_collisions(agent)
        reward += reward_collision

        # if collision with bounds or bound obstacle, revert to previous pos/ori
        agent.pos = self._clip_pos(agent.pos, collided_obs_id, agent.radius)

        # get internal and external state of agents (observation)
        internal_state = self._get_internal_state(agent, shortest_goal)
        external_state_laser, external_state_type = self._get_external_state(agent)
        observation = [internal_state, external_state_laser, external_state_type]

        # when training, do manual reset once if the agent is stuck in local optima or
        # if they are wandering in 2-obstacles
        if "train" in self.mode:
            self._check_resets(agent, previous_pos, collision_obstacle)

        return observation, reward, done, {}

    def _check_resets(self, agent, previous_pos, collision_obstacle):
        if self.step_count_same == 0:
            agent_x = previous_pos[0, 0]
            agent_y = previous_pos[1, 0]
            self.box = [agent_x - 1, agent_x + 1, agent_y - 1, agent_y + 1]
        elif self.step_count_same == self.max_step_count:
            self.reset_pos_necessary = True

        if self.step_count_obs == self.max_step_count:
            self.reset_pos_necessary = True

        if self.step_count == 4000:
            self.reset_pos_necessary = True

        """if self._in_local_optima(agent.pos):
            self.step_count_same += 1
        else:
            self.step_count_same = 0"""

        if collision_obstacle:
            self.step_count_obs += 1
        else:
            self.step_count_obs = 0

        #self.step_count += 1

        if self.reset_pos_necessary:
            self.step_count_same = 0
            self.step_count_obs = 0
            self.step_count = 0

    def _in_local_optima(self, pos):
        agent_x = pos[0, 0]
        agent_y = pos[1, 0]

        x_min = self.box[0]
        x_max = self.box[1]
        y_min = self.box[2]
        y_max = self.box[3]

        return x_min <= agent_x <= x_max and y_min <= agent_y <= y_max

    def _clip_pos(self, pos, collided_obs_id, radius):
        if collided_obs_id is None:
            return pos

        if collided_obs_id == -1:
            if (pos[0, 0] - radius) < self.bounds[0]:
                pos[0, 0] = self.bounds[0] + radius
            elif (pos[0, 0] + radius) > self.bounds[1]:
                pos[0, 0] = self.bounds[1] - radius

            if (pos[1, 0] - radius) < self.bounds[2]:
                pos[1, 0] = self.bounds[2] + radius
            elif (pos[1, 0] + radius) > self.bounds[3]:
                pos[1, 0] = self.bounds[3] - radius
        else:
            x_min = self.sim_state.obstacles[collided_obs_id].x
            x_max = x_min + self.sim_state.obstacles[collided_obs_id].width
            y_min = self.sim_state.obstacles[collided_obs_id].y
            y_max = y_min + self.sim_state.obstacles[collided_obs_id].height

            # clipping now only works for rectangular obstacles placed on bounds (as with the crossway examples)
            if x_min <= pos[0, 0] <= x_max:
                if pos[1, 0] < y_min and pos[1, 0] < y_max:
                    pos[1, 0] = y_min - radius
                if pos[1, 0] > y_min and pos[1, 0] > y_max:
                    pos[1, 0] = y_max + radius
            if y_min <= pos[1, 0] <= y_max:
                if pos[0, 0] < x_min and pos[0, 0] < x_max:
                    pos[0, 0] = x_min - radius
                if pos[0, 0] > x_min and pos[0, 0] > x_max:
                    pos[0, 0] = x_max + radius

        return pos

    @staticmethod
    def _get_internal_state(agent, goal):
        rotation_matrix_new = np.array([[math.cos(agent.orientation), -math.sin(agent.orientation)],
                                        [math.sin(agent.orientation), math.cos(agent.orientation)]])
        relative_pos_agent_to_goal = np.subtract(goal.pos, agent.pos)
        internal_state = np.matmul(np.linalg.inv(rotation_matrix_new), relative_pos_agent_to_goal)

        observation = np.array([
            internal_state[0, 0],
            internal_state[1, 0]
        ])

        return observation

    def _get_external_state(self, agent):
        laser_distances = [self.WORLD_BOUND * 2] * (self.sim_state.laser_amount + 1)
        types = []
        agent.laser_lines = []
        agent.type_colors = []
        max_view = 15

        start_point = agent.orientation - math.radians(90)
        increment = math.radians(180 / self.sim_state.laser_amount)
        for i in range(0, self.sim_state.laser_amount + 1):
            laser_ori = start_point + i * increment
            x_ori = math.cos(laser_ori)
            y_ori = math.sin(laser_ori)
            distance, x_end, y_end, type = self._get_first_crossed_object(agent, x_ori, y_ori, max_view)
            if type != 0:  # and distance <= max_view:
                laser_distances[i] = distance
            types.append(type)
            agent.laser_lines.append(np.array([x_end, y_end]))
            agent.type_colors.append(type)

        if len(agent.laser_history) == self.sim_state.laser_history_amount:
            agent.laser_history.pop(0)
            agent.type_history.pop(0)
        else:
            while len(agent.laser_history) < self.sim_state.laser_history_amount - 1:
                agent.laser_history.append(np.zeros(self.sim_state.laser_amount + 1))
                agent.type_history.append(np.zeros(self.sim_state.laser_amount + 1))
        agent.laser_history.append(np.array(laser_distances))
        agent.type_history.append(np.array(types))

        observation_laser = np.array(agent.laser_history)
        observation_type = np.array(agent.type_history)

        return observation_laser, observation_type

    def _get_first_crossed_object(self, current_agent, x_ori, y_ori, max_view):
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
        distance_to_object = 0

        while distance_to_object < max_view:
        #while True:
            distant_x += x_ori * iteration_step
            distant_y += y_ori * iteration_step
            distance_to_object = math.hypot(x_agent - distant_x, y_agent - distant_y)

            for agent in self.compare_agents:
                if current_agent.id == agent.id:
                    for goal in agent.goals:
                        goal_found = False
                        goal_type = goal.type
                        if goal_type == 1:
                            width = goal.box[0]
                            height = goal.box[1]
                            x_min = goal.pos[0, 0] - width / 2
                            y_min = goal.pos[1, 0] - height / 2
                            if self._point_in_rectangle(distant_x, distant_y,
                                                        x_min, y_min, width, height):
                                if distance_to_object < distance:
                                    goal_found = True
                        else:
                            if self._point_in_circle(distant_x, distant_y, goal.pos[0, 0], goal.pos[1, 0],
                                                     self.sim_state.goal_tolerance):
                                if distance_to_object < distance:
                                    goal_found = True
                        if goal_found:
                            distance = distance_to_object
                            x_end = distant_x
                            y_end = distant_y
                            collision = True
                            type = 0
                elif not agent.done:   # dont detect other agent if they are in done state
                    if self._point_in_circle(distant_x, distant_y, agent.pos[0, 0], agent.pos[1, 0], agent.radius):
                        distance = distance_to_object
                        x_end = distant_x
                        y_end = distant_y
                        collision = True
                        type = 1

            for obstacle in self.obstacles:
                if obstacle.contains(distant_x, distant_y):
                    if distance_to_object < distance:
                        distance = distance_to_object
                        x_end = distant_x
                        y_end = distant_y
                        collision = True
                        type = 2
                        break

            if collision:
                break
            else:
                if self._collision_bound(distant_x, distant_y,
                                         self.bounds[0], self.bounds[1], self.bounds[2], self.bounds[3],
                                         current_agent.radius):
                    if distance_to_object < distance:
                        distance = distance_to_object
                        x_end = distant_x
                        y_end = distant_y
                        type = 2
                    break

        if distance_to_object >= max_view and not collision:
            x_end = distant_x
            y_end = distant_y
            distance = max_view
            type = 2

        return distance, x_end, y_end, type

    def _detect_collisions(self, current_agent):
        reward = 0
        collision = False
        collision_obstacle = False
        collision_ids_agent = []
        collision_ids_obs = []
        collided_obs_id = None

        # detect collision with obstacles
        for obstacle in self.obstacles:
            if self._collision_circle_rectangle(obstacle.x, obstacle.y, obstacle.width, obstacle.height,
                                                current_agent.pos[0, 0], current_agent.pos[1, 0],
                                                current_agent.radius):
                collision_obstacle = True
                if obstacle.type == 0:
                    collision_ids_obs.append(obstacle.id)
                else:
                    collided_obs_id = obstacle.id
                if obstacle.id not in self.collision_ids_obs and obstacle.type == 0:
                    collision = True

                """if obstacle.type == 0:
                    collision = True
                else:
                    collided_obs_id = -1"""

        # detect collision with other agents
        if not collision:
            for agent in self.compare_agents:
                if current_agent.id != agent.id and not agent.done:  # only detect collision when agent not done
                    if self._collision_circle_circle(current_agent.pos[0, 0], current_agent.pos[1, 0],
                                                     current_agent.radius, agent.pos[0, 0], agent.pos[1, 0],
                                                     agent.radius):
                        collision_ids_agent.append(agent.id)
                        if agent.id not in self.collision_ids_agent:
                            collision = True

        # detect collision with world bounds
        if self._collision_bound(current_agent.pos[0, 0], current_agent.pos[1, 0],
                                 self.bounds[0], self.bounds[1], self.bounds[2], self.bounds[3],
                                 current_agent.radius):
            collided_obs_id = -1

        if collision:
            reward -= self.reward_collision

        if collided_obs_id is not None:
            reward -= self.reward_collision_clip

        self.collision_ids_agent = collision_ids_agent
        self.collision_ids_obs = collision_ids_obs

        return reward, collision_obstacle, collided_obs_id

    def reset(self):
        if "multi" not in self.mode:
            self.sim_state = copy.deepcopy(self.orig_sim_state)

        self._load_world()

        self.step_count_same = 0
        self.step_count_obs = 0

        agent = self.steering_agents[self.id]
        max_distance_to_goal = 0
        first = True
        shortest_goal = None
        for goal in agent.goals:
            distance_to_goal = self._calculate_distance_goal(agent.pos, goal)
            if first:
                max_distance_to_goal = distance_to_goal
                first = False
            if distance_to_goal <= max_distance_to_goal:
                shortest_goal = goal

        internal_state = self._get_internal_state(agent, shortest_goal)
        external_state_laser, external_state_type = self._get_external_state(agent)
        observation = [internal_state, external_state_laser, external_state_type]

        return observation

    def set_compare_state(self, agents):
        self.compare_agents = agents

    def get_agents(self):
        return self.steering_agents

    def render(self, mode='human'):
        pass

    def get_observation_space(self):
        return self.observation_space

    def get_action_space(self):
        return self.action_space

    def _calculate_distance_goal(self, agent_pos, goal):
        goal_type = goal.type
        # distance to point
        if goal_type == 0:
            return math.hypot(agent_pos[0, 0] - goal.pos[0, 0], agent_pos[1, 0] - goal.pos[1, 0])
        # distance to rectangle
        else:
            box = goal.box
            width = box[0]
            height = box[1]
            x_min = goal.pos[0, 0] - width / 2
            y_min = goal.pos[1, 0] - height / 2
            x_max = x_min + width
            y_max = y_min + height
            x = agent_pos[0, 0]
            y = agent_pos[1, 0]

            if x < x_min:
                if y < y_min:
                    return math.hypot(x_min - x, y_min - y)
                if y <= y_max:
                    return x_min - x
                return math.hypot(x_min - x, y_max - y)
            elif x <= x_max:
                if y < y_min:
                    return y_min - y
                if y <= y_max:
                    return 0
                return y - y_max
            else:
                if y < y_min:
                    return math.hypot(x_max - x, y_min - y)
                if y <= y_max:
                    return x - x_max
                return math.hypot(x_max - x, y_max - y)

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

        return distance <= r

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
    def _point_in_rectangle(x_p, y_p, x_r, y_r, width, height):
        return (x_r <= x_p <= (x_r + width)) and (y_r <= y_p <= (y_r + height))

    @staticmethod
    def _collision_bound(x_p, y_p, x_min, x_max, y_min, y_max, radius):
        return (x_p - radius) <= x_min or (x_p + radius) >= x_max or (y_p - radius) <= y_min or (y_p + radius) >= y_max
