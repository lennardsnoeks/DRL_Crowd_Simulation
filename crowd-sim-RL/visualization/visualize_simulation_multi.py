import sys
import pygame
import math
import copy
import numpy as np
from pygame.locals import *
from ray.rllib import MultiAgentEnv
from ray.rllib.agents import Trainer
from utils.steerbench_parser import SimulationState
from visualization.visualize_config import SIM_COLORS


class VisualizationSimMulti:

    def __init__(self, sim_state: SimulationState, trainer: Trainer):
        pygame.init()

        self.FPS_FONT = pygame.font.SysFont("Verdana", 12)
        self.GOLDENROD = pygame.Color("goldenrod")

        self.goals_visible = True
        self.lasers_visible = False

        self.framerate = 30

        self.offset = 0.0
        self.zoom_factor = 10
        self.background_color = SIM_COLORS['white']
        self.border_width = 1
        self.border_color = SIM_COLORS['light gray']
        self.obstacle_color = SIM_COLORS['gray']
        self.sim_state = copy.deepcopy(sim_state)
        self.trainer = trainer

    def run(self):
        pygame.init()
        self.initialize_screen()
        clock = pygame.time.Clock()

        config = {
            "sim_state": self.sim_state,
            "agent_id": 0,
            "mode": "sim"
        }

        env = MultiAgentEnv(config)
        observations = env.reset()
        done = False
        prev_actions = {}
        prev_rewards = {}
        for i in range(0, self.sim_state.agents):
            prev_rewards[i] = 0
            prev_actions[i] = [0.0, 0.0]
        state = self.trainer.get_policy().get_initial_state()

        while not done:
            dt = clock.tick(self.framerate)

            self.process_events()

            actions = {}
            action_rescales = {}

            for i in range(0, self.sim_state.agents):
                linear_vel, angular_vel = self.trainer.compute_action(observations[i],
                                                                      state=state,
                                                                      prev_action=prev_actions[i],
                                                                      prev_reward=prev_rewards[i],
                                                                      explore=False)
                scale = dt / 1000
                action_rescales[i] = [linear_vel * scale, angular_vel * scale]
                actions[i] = [linear_vel, angular_vel]

            observations, rewards, dones, info = env.step(action_rescales)

            prev_actions = actions
            prev_rewards = rewards

            self.simulation_update()

            self.show_fps(self.screen, clock)
            self.show_size(self.screen)

            pygame.display.flip()

    def show_fps(self, window, clock):
        fps_overlay = self.FPS_FONT.render(str(round(clock.get_fps(), 2)) + " fps", True, self.GOLDENROD)
        window.blit(fps_overlay, (0, 0))

    def show_size(self, window):
        x_size = self.sim_state.clipped_bounds[1] - self.sim_state.clipped_bounds[0]
        y_size = self.sim_state.clipped_bounds[3] - self.sim_state.clipped_bounds[2]
        size_overlay = self.FPS_FONT.render(str(x_size) + " x " + str(y_size), True, self.GOLDENROD)
        window.blit(size_overlay, (0, 14))

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
                if not (goal.tolist() in unique_goals):
                    unique_goals.append(goal.tolist())

        for goal in unique_goals:
            goal = np.array(goal)
            pygame.draw.circle(self.screen, SIM_COLORS['green'],
                               (int(goal[0, 0] * self.zoom_factor), int(goal[1, 0] * self.zoom_factor)),
                               self.zoom_factor, 0)

    def draw_lasers(self):
        for agent in self.sim_state.agents:
            agent_pos_x = agent.pos[0, 0] * self.zoom_factor
            agent_pos_y = agent.pos[1, 0] * self.zoom_factor

            for i in range(0, len(agent.laser_lines)):
                laser_end_x = agent.laser_lines[i][0] * self.zoom_factor
                laser_end_y = agent.laser_lines[i][1] * self.zoom_factor
                pygame.draw.line(self.screen, SIM_COLORS['white'],
                                 (agent_pos_x, agent_pos_y), (laser_end_x, laser_end_y), 1)

    def quit(self):
        sys.exit()
