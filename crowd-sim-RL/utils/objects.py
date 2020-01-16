import numpy as np


class Vec2:
    x = 0.0
    y = 0.0
    matrix_form = []

    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.matrix_form = np.array([[self.x], [self.y]])


class Obstacle:
    x = 0.0
    y = 0.0
    width = 0.0
    height = 0.0

    def __init__(self, width, height, x, y):
        self.width = width
        self.height = height
        self.x = x
        self.y = y

    def contains(self, pos):
        return (self.x <= pos[0, 0] <= (self.x + self.width)) and (self.y <= pos[1, 0] <= (self.y + self.height))


class Agent:
    pos = None
    orientation = 0.0
    radius = 0.0
    goals = []
    initial_speed = None
    fov = 0.0
    id = 0.0
    color = []
    laser_history = []

    def __init__(self, pos, radius, orientation, goals, initial_speed, fov, id, color):
        self.pos = pos
        self.radius = radius
        self.orientation = orientation
        self.goals = goals
        self.initial_speed = initial_speed
        self.fov = fov
        self.id = id
        self.color = color

