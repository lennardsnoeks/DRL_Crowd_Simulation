import sys
import pygame
import math
import copy
import numpy as np
from pygame.locals import *
from pygame import Color

from utils.steerbench_parser import SimulationState

SIM_COLORS = {
    'aqua': Color(0, 255, 255),
    'black': Color(0, 0, 0),
    'blue': Color(0, 0, 255),
    'fuchsia': Color(255, 0, 255),
    'gray': Color(128, 128, 128),
    'light gray': Color(50, 50, 50),
    'green': Color(0, 128, 0),
    'lime': Color(0, 255, 0),
    'maroon': Color(128, 0, 0),
    'navy blue': Color(0, 0, 128),
    'olive': Color(128, 128, 0),
    'purple': Color(128, 0, 128),
    'red': Color(255, 0, 0),
    'silver': Color(192, 192, 192),
    'teal': Color(0, 128, 128),
    'white': Color(255, 255, 255),
    'yellow': Color(255, 255, 0)
}


class Visualization:
    BOUNDARIES = 0, 0, 600, 600

    def __init__(self, sim_state: SimulationState):
        pygame.init()

        self.goals_visible = True

        self.offset = 0.0
        self.zoom_factor = 10
        self.background_color = SIM_COLORS['white']
        self.border_width = 1
        self.border_color = SIM_COLORS['light gray']
        self.obstacle_color = SIM_COLORS['gray']
        self.sim_state = copy.deepcopy(sim_state)

        self.clock = pygame.time.Clock()
        self.paused = False
        self.time = 0
        self.time_passed = 0
        self.timer_interval = 10

        self.initialize_screen()

    def run(self):
        pygame.init()

        while True:
            self.time_passed = self.clock.tick()

            self.process_events()

            if not self.paused:
                self.time += self.time_passed
                if self.time > self.timer_interval:
                    self.time -= self.timer_interval
                    self.simulation_update()

            pygame.display.flip()

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
                               (int(goal[0, 0] * self.zoom_factor), int(goal[1,0] * self.zoom_factor)),
                               self.zoom_factor, 0)

    def update_agents(self, updated_agents):
        copy_agents = []
        for agent in updated_agents:
            copy_agents.append(copy.copy(agent))
        self.sim_state.agents = copy_agents

        #ev = pygame.event.Event(pygame.USEREVENT, {'data': copy_agents})
        #pygame.event.post(ev)

    def quit(self):
        sys.exit()
