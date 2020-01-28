import sys
import pygame
import math
import copy
import numpy as np
from pygame.locals import *
from ray.rllib.agents import Trainer

from utils.steerbench_parser import SimulationState
from visualization.visualize_config import SIM_COLORS


class VisualizationSim:

    def __init__(self, sim_state: SimulationState, trainer: Trainer):
        pygame.init()

        self.goals_visible = True
        self.lasers_visible = True

        self.framerate = 30

        self.offset = 0.0
        self.zoom_factor = 10
        self.background_color = SIM_COLORS['white']
        self.border_width = 1
        self.border_color = SIM_COLORS['light gray']
        self.obstacle_color = SIM_COLORS['gray']
        self.sim_state = copy.deepcopy(sim_state)
        self.trainer = trainer

        self.paused = False
        self.time = 0
        self.time_passed = 0
        self.timer_interval = 10
        self.active = True

    def retrieve_action(self, current_agent):

        for agent in self.sim_state.agents:
            if agent.id == current_agent.id:
                shortest_goal = self._get_shortest_goal(agent)
                internal_state = self._get_internal_state(agent, shortest_goal)
                external_state = self._get_external_state(agent)
                observation = [internal_state, external_state]
                self.trainer.compute_action(observation, state=None, prev_action=None, prev_reward=None, info=None, policy_id=DEFAULT_POLICY_ID, full_fetch=False)

    @staticmethod
    def _get_shortest_goal(agent):
        max_goal_distance = 1000000
        shortest_goal = None

        for goal in agent.goals:
            distance_to_goal = math.sqrt((agent.pos[0, 0] - goal[0, 0]) ** 2 + (agent.pos[1, 0] - goal[1, 0]) ** 2)
            if distance_to_goal < max_goal_distance:
                shortest_goal = goal

        return shortest_goal

    @staticmethod
    def _get_internal_state(agent, goal):
        rotation_matrix_new = np.array([[math.cos(agent.orientation), -math.sin(agent.orientation)],
                                        [math.sin(agent.orientation), math.cos(agent.orientation)]])
        relative_pos_agent_to_goal = np.subtract(goal, agent.pos)
        internal_state = np.matmul(np.linalg.inv(rotation_matrix_new), relative_pos_agent_to_goal)

        observation = np.array([
            internal_state[0, 0],
            internal_state[1, 0]
        ])

        return observation

    def _get_external_state(self, agent):
        laser_distances = []
        agent.laser_lines = []

        start_point = agent.orientation - math.radians(90)
        increment = math.radians(180 / self.sim_state.laser_amount)
        for i in range(0, self.sim_state.laser_amount + 1):
            laser_ori = start_point + i * increment
            x_ori = math.cos(laser_ori)
            y_ori = math.sin(laser_ori)
            distance, x_end, y_end = self._get_first_crossed_object(agent.pos[0, 0], agent.pos[1, 0], x_ori, y_ori)
            laser_distances.append(distance)
            agent.laser_lines.append(np.array([x_end, y_end]))

        if len(agent.laser_history) == self.sim_state.laser_history_amount:
            agent.laser_history.pop(0)
        else:
            while len(agent.laser_history) < self.sim_state.laser_history_amount - 1:
                agent.laser_history.append(np.zeros(self.sim_state.laser_amount + 1))
        agent.laser_history.append(np.array(laser_distances))

        observation = np.array(agent.laser_history)

        return observation

    def _get_first_crossed_object(self, x_agent, y_agent, x_ori, y_ori):
        distance = 1000000
        x_end = 1000000
        y_end = 1000000
        iteration_step = 0.05
        collision = False

        distant_x = x_agent
        distant_y = y_agent
        while True:
            distant_x += x_ori * iteration_step
            distant_y += y_ori * iteration_step
            for agent in self.sim_state.agents:
                if x_agent != agent.pos[0, 0] and y_agent != agent.pos[1, 0]:
                    if self._point_in_circle(distant_x, distant_y, agent.pos[0, 0], agent.pos[1, 0], agent.radius):
                        distance_agent = math.sqrt((x_agent - distant_x) ** 2 + (y_agent - distant_y) ** 2)
                        distance = distance_agent
                        x_end = distant_x
                        y_end = distant_y
                        collision = True
                        break

            for obstacle in self.sim_state.obstacles:
                if obstacle.contains(distant_x, distant_y):
                    distance_obstacle = math.sqrt((x_agent - distant_x) ** 2 + (y_agent - distant_y) ** 2)
                    if distance_obstacle < distance:
                        distance = distance_obstacle
                        x_end = distant_x
                        y_end = distant_y
                        collision = True
                        break

            if collision:
                break
            else:
                if self._collision_bound(distant_x, distant_y,
                                         self.sim_state.bounds[0], self.sim_state.bounds[1],
                                         self.sim_state.bounds[2], self.sim_state.bounds[3]):
                    distance_bound = math.sqrt((x_agent - distant_x) ** 2 + (y_agent - distant_y) ** 2)
                    if distance_bound < distance:
                        distance = distance_bound
                        x_end = distant_x
                        y_end = distant_y
                    break

        return distance, x_end, y_end

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

    def run(self):
        pygame.init()

        self.initialize_screen()

        clock = pygame.time.Clock()

        dt = 0

        while self.active:
            dt = clock.tick(self.framerate)

            self.process_events()



            pygame.display.flip()

    def stop(self):
        self.active = False
        self.quit()

    def initialize_screen(self):
        self.width = self.sim_state.clipped_bounds[1] - self.sim_state.clipped_bounds[0]
        self.height = self.sim_state.clipped_bounds[3] - self.sim_state.clipped_bounds[2]
        self.screen = pygame.display.set_mode((int(self.width) * self.zoom_factor, int(self.height) * self.zoom_factor),
                                              HWSURFACE | DOUBLEBUF | RESIZABLE, 32)

        self.field = pygame.Rect(self.sim_state.clipped_bounds[0],
                                 self.sim_state.clipped_bounds[2],
                                 self.width * self.zoom_factor,
                                 self.height * self.zoom_factor)

        self.internal_field = pygame.Rect(self.field.left + self.border_width,
                                          self.field.top + self.border_width,
                                          self.field.width - self.border_width * 2,
                                          self.field.height - self.border_width * 2)

    def simulation_update(self):
        self.draw()

    def process_events(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.quit()
            if event.type == pygame.USEREVENT:
                self.sim_state.agents = event.data

    def draw(self):
        self.draw_background()
        self.draw_obstacles()
        self.draw_agents()
        if self.goals_visible:
            self.draw_goals()
        if self.lasers_visible:
            self.draw_lasers()

    def draw_background(self):
        pygame.draw.rect(self.screen, SIM_COLORS['gray'], self.field)

    def draw_obstacles(self):
        for obstacle in self.sim_state.obstacles:
            x_min = obstacle.x * self.zoom_factor
            y_min = obstacle.y * self.zoom_factor
            x_max = (obstacle.x + obstacle.width) * self.zoom_factor
            y_max = (obstacle.y + obstacle.height) * self.zoom_factor

            # draw obstacle
            pygame.draw.rect(self.screen, SIM_COLORS['light gray'],
                             (x_min, y_min, obstacle.width * self.zoom_factor, obstacle.height * self.zoom_factor))

            # draw border
            pygame.draw.line(self.screen, SIM_COLORS['black'], (x_min, y_min), (x_max, y_min), 1)
            pygame.draw.line(self.screen, SIM_COLORS['black'], (x_min, y_min), (x_min, y_max), 1)
            pygame.draw.line(self.screen, SIM_COLORS['black'], (x_max, y_min), (x_max, y_max), 1)
            pygame.draw.line(self.screen, SIM_COLORS['black'], (x_min, y_max), (x_max, y_max), 1)

    def draw_agents(self):
        for agent in self.sim_state.agents:
            color = Color(agent.color[0], agent.color[1], agent.color[2])
            agent_pos_x = agent.pos[0, 0] * self.zoom_factor
            agent_pos_y = agent.pos[1, 0] * self.zoom_factor
            pygame.draw.circle(self.screen, color,
                               (int(agent_pos_x), int(agent_pos_y)), int(agent.radius * self.zoom_factor), 0)

            orientation_2d = np.array([math.cos(agent.orientation), math.sin(agent.orientation)])
            point2_x = (agent_pos_x + (orientation_2d[0] * agent.radius * self.zoom_factor))
            point2_y = (agent_pos_y + (orientation_2d[1] * agent.radius * self.zoom_factor))

            pygame.draw.line(self.screen, SIM_COLORS['white'], (agent_pos_x, agent_pos_y), (point2_x, point2_y), 1)

    def draw_goals(self):
        unique_goals = []
        for agent in self.sim_state.agents:
            for goal in agent.goals:
                if not (goal in unique_goals):
                    unique_goals.append(goal)

        for goal in unique_goals:
            pygame.draw.circle(self.screen, SIM_COLORS['green'],
                               (int(goal[0, 0] * self.zoom_factor), int(goal[1, 0] * self.zoom_factor)),
                               self.zoom_factor, 0)

    def draw_lasers(self):
        for agent in self.sim_state.agents:
            agent_pos_x = agent.pos[0, 0] * self.zoom_factor
            agent_pos_y = agent.pos[1, 0] * self.zoom_factor

            for laser in agent.laser_lines:
                laser_end_x = laser[0] * self.zoom_factor
                laser_end_y = laser[1] * self.zoom_factor
                pygame.draw.line(self.screen, SIM_COLORS['white'],
                                 (agent_pos_x, agent_pos_y), (laser_end_x, laser_end_y), 1)

    def update_agents(self, updated_agents):
        copy_agents = []
        for agent in updated_agents:
            copy_agents.append(copy.copy(agent))
        self.sim_state.agents = copy_agents

        try:
            ev = pygame.event.Event(pygame.USEREVENT, {'data': copy_agents})
            pygame.event.post(ev)
        except pygame.error:
            pass

    def quit(self):
        sys.exit()
