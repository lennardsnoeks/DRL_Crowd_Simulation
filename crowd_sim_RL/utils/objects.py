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


class Agent:
    pos = None
    radius = 0.0
    gsd = 0.0
    goals = []
    initial_speed = None
    fov = 0.0
    id = 0.0
    color = []

    def __init__(self, pos, radius, gsd, goals, initial_speed, fov, id, color):
        self.pos = pos
        self.radius = radius
        self.gsd = gsd
        self.goals = goals
        self.initial_speed = initial_speed
        self.fov = fov
        self.id = id
        self.color = color

