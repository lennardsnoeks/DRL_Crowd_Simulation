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

    def contains(self, x, y):
        return (self.x <= x <= (self.x + self.width)) and (self.y <= y <= (self.y + self.height))


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
    type_history = []
    laser_lines = []

    def __init__(self, pos, radius, orientation, goals, initial_speed, fov, id, color):
        self.pos = pos
        self.radius = radius
        self.orientation = orientation
        self.goals = goals
        self.initial_speed = initial_speed
        self.fov = fov
        self.id = id
        self.color = color
