import sys
from time import sleep

import pygame
import math
import copy
import numpy as np
from pygame.locals import *
from ray.rllib.agents import Trainer

from crowd_sim_RL.envs import MultiAgentCentralized
from utils.steerbench_parser import SimulationState
from visualization.color_config import SIM_COLORS


class VisualizationSimMultiCentralized:

    def __init__(self, sim_state: SimulationState, trainer: Trainer, zoom_factor):
        pygame.init()

        self.FPS_FONT = pygame.font.SysFont("Verdana", 11)
        self.GOLDENROD = pygame.Color("goldenrod")

        self.goals_visible = True
        self.lasers_visible = False
        self.show_path = True
        self.color_lasers = True

        self.framerate = 30

        self.offset = 0.0
        self.zoom_factor = zoom_factor
        self.background_color = SIM_COLORS['white']
        self.border_width = 1
        self.border_color = SIM_COLORS['light gray']
        self.obstacle_color = SIM_COLORS['gray']
        self.sim_state = copy.deepcopy(sim_state)
        self.trainer = trainer

        self.history_all_agents = [None] * len(sim_state.agents)
        for agent in self.sim_state.agents:
            previous_positions = [agent.pos]
            self.history_all_agents[agent.id] = previous_positions

        self.initialize_screen()

    def run(self):
        config = {
            "sim_state": self.sim_state,
            "agent_id": 0,
            "mode": "sim"
        }

        env = MultiAgentCentralized(config)
        observation = env.reset()
        done = False
        prev_action = np.zeros_like(env.action_space.sample())
        prev_reward = 0
        state = self.trainer.get_policy().get_initial_state()

        self.simulation_update()
        pygame.display.flip()
        sleep(5)

        actions = self.trainer.compute_action(observation,
                                              state=state,
                                              prev_action=prev_action,
                                              prev_reward=prev_reward)

        scale = 0.033

        action_rescales = [action * scale for action in actions]

        observation, reward, done, info = env.step(action_rescales)

        prev_action = actions
        prev_reward = reward

        clock = pygame.time.Clock()

        while True:
            dt = clock.tick(self.framerate)

            self.process_events()

            if not done:
                actions = self.trainer.compute_action(observation,
                                                      state=state,
                                                      prev_action=prev_action,
                                                      prev_reward=prev_reward)

                scale = dt / 1000

                action_rescales = [action * scale for action in actions]

                observation, reward, done, info = env.step(action_rescales)

                prev_action = actions
                prev_reward = reward

                agents = env.get_agents()
                self.sim_state.agents = agents

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
            if obstacle.type == -1:
                color = SIM_COLORS['light gray']
            else:
                color = SIM_COLORS['obs']
            pygame.draw.rect(self.screen, color,
                             (x_min, y_min, obstacle.width * self.zoom_factor, obstacle.height * self.zoom_factor))

            # draw border
            pygame.draw.line(self.screen, color, (x_min, y_min), (x_max, y_min), 1)
            pygame.draw.line(self.screen, color, (x_min, y_min), (x_min, y_max), 1)
            pygame.draw.line(self.screen, color, (x_max, y_min), (x_max, y_max), 1)
            pygame.draw.line(self.screen, color, (x_min, y_max), (x_max, y_max), 1)

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

            if self.show_path:
                self.history_all_agents[agent.id].append(agent.pos)
                history = self.history_all_agents[agent.id]

                for i in range(0, len(history)):
                    if i == len(history) - 1:
                        break
                    pos = history[i]
                    pos2 = history[i + 1]
                    pygame.draw.line(self.screen, color,
                                     (pos[0, 0] * self.zoom_factor, pos[1, 0] * self.zoom_factor),
                                     (pos2[0, 0] * self.zoom_factor, pos2[1, 0] * self.zoom_factor))

    def draw_goals(self):
        unique_goals = []
        for agent in self.sim_state.agents:
            for goal in agent.goals:
                if not (goal in unique_goals):
                    unique_goals.append(goal)

        for goal in unique_goals:
            if goal.type == 1:
                width = goal.box[0]
                height = goal.box[1]
                min_goal_x = goal.pos[0, 0] - width / 2
                min_goal_y = goal.pos[1, 0] - height / 2
                pygame.draw.rect(self.screen, SIM_COLORS['green'],
                                 (int(min_goal_x * self.zoom_factor),
                                  int(min_goal_y * self.zoom_factor),
                                  width * self.zoom_factor,
                                  height * self.zoom_factor))
            else:
                pygame.draw.circle(self.screen, SIM_COLORS['green'],
                                   (int(goal.pos[0, 0] * self.zoom_factor), int(goal.pos[1, 0] * self.zoom_factor)),
                                   self.zoom_factor, 0)

    def draw_lasers(self):
        for agent in self.sim_state.agents:
            agent_pos_x = agent.pos[0, 0] * self.zoom_factor
            agent_pos_y = agent.pos[1, 0] * self.zoom_factor

            i = 0
            for laser in agent.laser_lines:
                laser_end_x = laser[0] * self.zoom_factor
                laser_end_y = laser[1] * self.zoom_factor

                type = agent.type_history[self.sim_state.laser_history_amount][i]
                if self.color_lasers and type == 0:
                    color = SIM_COLORS['green']
                elif self.color_lasers and type == 1:
                    color = SIM_COLORS['red']
                else:
                    color = SIM_COLORS['white']
                pygame.draw.line(self.screen, color,
                                 (agent_pos_x, agent_pos_y), (laser_end_x, laser_end_y), 1)
                i += 1

    def quit(self):
        sys.exit()
