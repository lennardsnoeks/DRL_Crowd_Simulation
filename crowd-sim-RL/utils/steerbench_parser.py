import xml.etree.ElementTree as ET
import numpy as np
import random
import math
from utils.objects import Obstacle, Agent


rainbow = [
    [148, 0, 211],
    [255, 0, 0],
    [75, 0, 130],
    [255, 127, 0],
    [0, 0, 255],
    [0, 255, 0],
    [255, 0, 0]
]


class SimulationState:

    def __init__(self):
        self.agents = []
        self.obstacles = []
        self.bounds = []
        self.clipped_bounds = []

        self.goal_tolerance = 2
        self.laser_history_amount = 3
        self.laser_amount = 10

    def shift_center(self):
        shift = np.array([[-self.clipped_bounds[0]], [-self.clipped_bounds[2]]])
        for agent in self.agents:
            agent.pos = np.add(agent.pos, shift)
            for i in range(len(agent.goals)):
                agent.goals[i] = np.add(agent.goals[i], shift)

        for obstacle in self.obstacles:
            obstacle.x = obstacle.x + shift[0, 0]
            obstacle.y = obstacle.y + shift[1, 0]

        self.bounds = [
            self.bounds[0] + shift[0, 0],
            self.bounds[1] + shift[0, 0],
            self.bounds[2] + shift[1, 0],
            self.bounds[3] + shift[1, 0]
        ]

        self.clipped_bounds = [
            self.clipped_bounds[0] + shift[0, 0],
            self.clipped_bounds[1] + shift[0, 0],
            self.clipped_bounds[2] + shift[1, 0],
            self.clipped_bounds[3] + shift[1, 0]
        ]

    def clip_bounds(self):
        margin = 0
        clipped_bounds = [10000000, -10000000, 10000000, -10000000]

        for agent in self.agents:
            clipped_bounds = [
                min(clipped_bounds[0], agent.pos[0, 0]),
                max(clipped_bounds[1], agent.pos[0, 0]),
                min(clipped_bounds[2], agent.pos[1, 0]),
                max(clipped_bounds[3], agent.pos[1, 0])
            ]
            for goal in agent.goals:
                clipped_bounds = [
                    min(clipped_bounds[0], goal[0, 0]),
                    max(clipped_bounds[1], goal[0, 0]),
                    min(clipped_bounds[2], goal[1, 0]),
                    max(clipped_bounds[3], goal[1, 0])
                ]

        for obstacle in self.obstacles:
            clipped_bounds = [
                min(clipped_bounds[0], obstacle.x),
                max(clipped_bounds[1], obstacle.x + obstacle.width),
                min(clipped_bounds[2], obstacle.y),
                max(clipped_bounds[3], obstacle.y + obstacle.height)
            ]

        self.clipped_bounds = [
            clipped_bounds[0] - (clipped_bounds[1] - clipped_bounds[0]) * margin,
            clipped_bounds[1] + (clipped_bounds[1] - clipped_bounds[0]) * margin,
            clipped_bounds[2] - (clipped_bounds[3] - clipped_bounds[2]) * margin,
            clipped_bounds[3] + (clipped_bounds[3] - clipped_bounds[2]) * margin
        ]

    def move_agents_from_obstacles(self):
        for agent in self.agents:
            for obstacle in self.obstacles:
                if obstacle.contains(agent.pos[0, 0], agent.pos[1, 0]):
                    agent.pos = np.array([[obstacle.x + obstacle.width + 1],
                                          [obstacle.y + obstacle.height + 1]])

            for goal_num, goal in enumerate(agent.goals):
                for obstacle in self.obstacles:
                    if obstacle.contains(goal[0, 0], goal[1, 0]):
                        agent.goals[goal_num] = np.array([[obstacle.x + obstacle.width + 1],
                                                          [obstacle.y + obstacle.height + 1]])


class XMLSimulationState:

    def __init__(self, filename):
        self.simulation_state = SimulationState()
        self.namespace = {
            'steerbench': 'http://www.magix.ucla.edu/steerbench'
        }
        self.fov = 5
        self.rainbow_index = 0

        self.parse_xml(filename)

    def parse_xml(self, filename):
        tree = ET.parse(filename)
        root = tree.getroot()
        self.parse_header(root.find('steerbench:header', self.namespace))
        self.parse_obstacles(root)
        self.parse_agents(root)

        self.simulation_state.clip_bounds()
        self.simulation_state.shift_center()
        self.simulation_state.move_agents_from_obstacles()

    def get_sim_state(self):
        return self.simulation_state

    def parse_header(self, header_elem):
        bounds = self.parse_bounds(header_elem.find('steerbench:worldBounds', self.namespace))
        self.simulation_state.bounds = [bounds[0], bounds[1], bounds[4], bounds[5]]

    def parse_agents(self, root):
        agents = root.findall('steerbench:agent', self.namespace)
        for agent in agents:
            self.parse_agent(agent)

        agent_regions = root.findall('steerbench:agentRegion', self.namespace)
        for region in agent_regions:
            self.parse_agent_region(region)

    def parse_agent(self, element):
        initial_config = element.find('steerbench:initialConditions', self.namespace)
        pos = self.parse_vector(initial_config.find('steerbench:position', self.namespace),
                                self.simulation_state.bounds)
        direction = self.parse_vector(initial_config.find('steerbench:direction', self.namespace), [-1, 1, -1, 1])

        goal_config = element.find('steerbench:goalSequence', self.namespace)
        goals = []
        for target in goal_config.findall('steerbench:seekStaticTarget', self.namespace):
            goals.append(self.parse_vector(target.find('steerbench:targetLocation', self.namespace),
                                           self.simulation_state.bounds))

        speed = float(initial_config.find('steerbench:speed', self.namespace).text)

        orientation = math.atan2(direction[1, 0], direction[0, 0])

        color = rainbow[self.rainbow_index % 7]
        self.rainbow_index += 1

        self.simulation_state.agents.append(
            Agent(pos=pos, radius=0.5, orientation=orientation, goals=goals, initial_speed=speed, fov=self.fov,
                  id=len(self.simulation_state.agents), color=color)
        )

    def parse_agent_region(self, element):
        num = int(element.find('steerbench:numAgents', self.namespace).text)
        bounds = self.parse_bounds(element.find('steerbench:regionBounds', self.namespace))
        initial_config = element.find('steerbench:initialConditions', self.namespace)

        color = rainbow[self.rainbow_index % 7]
        self.rainbow_index += 1

        for i in range(num):
            pos = np.array([[random.uniform(bounds[0], bounds[1])], [random.uniform(bounds[4], bounds[5])]])
            direction = self.parse_vector(initial_config.find('steerbench:direction', self.namespace), [-1, 1, -1, 1])

            goal_config = element.find('steerbench:goalSequence', self.namespace)
            goals = []
            for target in goal_config.findall('steerbench:seekStaticTarget', self.namespace):
                goals.append(self.parse_vector(target.find('steerbench:targetLocation', self.namespace),
                                               [bounds[0], bounds[1], bounds[4], bounds[5]]))

            speed = float(initial_config.find('steerbench:speed', self.namespace).text)

            orientation = math.atan2(direction[1, 0], direction[0, 0])

            self.simulation_state.agents.append(
                Agent(pos=pos, radius=0.5, orientation=orientation, goals=goals, initial_speed=speed, fov=self.fov,
                      id=len(self.simulation_state.agents), color=color)
            )

    def parse_vector(self, element, bounds):
        if element.find('steerbench:random', self.namespace) is not None:
            return np.array([
                [random.uniform(bounds[0], bounds[1])],
                [random.uniform(bounds[2], bounds[3])]
            ])
        else:
            return np.array([
                [float(element.find('steerbench:x', self.namespace).text)],
                [float(element.find('steerbench:z', self.namespace).text)]
            ])

    def parse_obstacles(self, root):
        obstacles = root.findall('steerbench:obstacle', self.namespace)
        for obstacle in obstacles:
            self.parse_obstacle(obstacle)

        obstacle_regions = root.findall('steerbench:obstacleRegion', self.namespace)
        for region in obstacle_regions:
            self.parse_obstacle_region(region)

    def parse_obstacle(self, element):
        bounds = self.parse_bounds(element)
        self.simulation_state.obstacles.append(
            Obstacle(bounds[1] - bounds[0], bounds[5] - bounds[4], bounds[0], bounds[4]))

    def parse_obstacle_region(self, region_elem):
        bounds = self.parse_bounds(region_elem.find('steerbench:regionBounds', self.namespace))
        num = int(region_elem.find('steerbench:numObstacles', self.namespace).text)
        size = float(region_elem.find('steerbench:obstacleSize', self.namespace).text)
        for i in range(num):
            self.simulation_state.obstacles.append(
                Obstacle(size, size, random.uniform(bounds[0], bounds[1]), random.uniform(bounds[4], bounds[5])))

    def parse_bounds(self, bounds_elem):
        bounds = [min(float(bounds_elem.find('steerbench:xmin', self.namespace).text),
                      float(bounds_elem.find('steerbench:xmax', self.namespace).text)),
                  max(float(bounds_elem.find('steerbench:xmin', self.namespace).text),
                      float(bounds_elem.find('steerbench:xmax', self.namespace).text)),
                  min(float(bounds_elem.find('steerbench:ymin', self.namespace).text),
                      float(bounds_elem.find('steerbench:ymax', self.namespace).text)),
                  max(float(bounds_elem.find('steerbench:ymin', self.namespace).text),
                      float(bounds_elem.find('steerbench:ymax', self.namespace).text)),
                  min(float(bounds_elem.find('steerbench:zmin', self.namespace).text),
                      float(bounds_elem.find('steerbench:zmax', self.namespace).text)),
                  max(float(bounds_elem.find('steerbench:zmin', self.namespace).text),
                      float(bounds_elem.find('steerbench:zmax', self.namespace).text)),
                  ]

        return bounds
