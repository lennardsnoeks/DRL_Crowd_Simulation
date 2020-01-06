import sys
import pygame
import math
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
    SCREEN_WIDTH, SCREEN_HEIGHT = 800, 800
    BOUNDARIES = 0, 0, 600, 600

    def __init__(self, sim_state: SimulationState):
        pygame.init()

        self.offset = 0.0
        self.zoom_factor = 1.0
        self.background_color = SIM_COLORS['white']
        self.border_width = 1
        self.border_color = SIM_COLORS['light gray']
        self.obstacle_color = SIM_COLORS['gray']
        self.sim_state = sim_state

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
        self.screen = pygame.display.set_mode((self.SCREEN_WIDTH, self.SCREEN_HEIGHT),
                                              HWSURFACE | DOUBLEBUF | RESIZABLE, 32)

        self.field = pygame.Rect(self.sim_state.bounds[0],
                                 self.sim_state.bounds[2],
                                 self.BOUNDARIES[2] * self.zoom_factor,
                                 self.BOUNDARIES[3] * self.zoom_factor)

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

    def draw(self):
        self.draw_background()
        self.draw_obstacles()
        self.draw_agents()

    def draw_background(self):
        pygame.draw.rect(self.screen, SIM_COLORS['light gray'], [0, 0, self.SCREEN_WIDTH, self.SCREEN_HEIGHT])
        pygame.draw.rect(self.screen, self.background_color, self.field)

    def draw_obstacles(self):
        for obstacle in self.sim_state.obstacles:
            x_min = obstacle.x
            y_min = obstacle.y
            x_max = obstacle.x + obstacle.width
            y_max = obstacle.y + obstacle.height

            # draw obstacle
            pygame.draw.rect(self.screen, SIM_COLORS['gray'], (x_min, y_min, obstacle.width, obstacle.height))

            # draw border
            pygame.draw.line(self.screen, SIM_COLORS['black'], (x_min, y_min), (x_max, y_min), 1)
            pygame.draw.line(self.screen, SIM_COLORS['black'], (x_min, y_min), (x_min, y_max), 1)
            pygame.draw.line(self.screen, SIM_COLORS['black'], (x_max, y_min), (x_max, y_max), 1)
            pygame.draw.line(self.screen, SIM_COLORS['black'], (x_min, y_max), (x_max, y_max), 1)

    def draw_agents(self):
        for agent in self.sim_state.agents:
            color = Color(agent.color[0], agent.color[1], agent.color[2])
            agent_pos_x = agent.pos[0, 0]
            agent_pos_y = agent.pos[1, 0]
            print(agent_pos_x, agent_pos_y)
            pygame.draw.circle(self.screen, color, (int(agent_pos_x), int(agent_pos_y)), int(agent.radius), 0)

            orientation_2d = np.array([math.cos(agent.orientation), math.sin(agent.orientation)])
            point2_x = (agent_pos_x + orientation_2d[0]) * agent.radius
            point2_y = (agent_pos_y + orientation_2d[1]) * agent.radius

            pygame.draw.line(self.screen, SIM_COLORS['white'], (agent_pos_x, agent_pos_y), (point2_x, point2_y), 2)

    def quit(self):
        sys.exit()
