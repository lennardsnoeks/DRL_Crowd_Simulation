class Obstacle:
    x = 0.0
    y = 0.0
    width = 0.0
    height = 0.0
    type = 0

    def __init__(self, width, height, x, y, type, id):
        self.width = width
        self.height = height
        self.x = x
        self.y = y
        self.type = type
        self.id = id

    def contains(self, x, y):
        return (self.x <= x <= (self.x + self.width)) and (self.y <= y <= (self.y + self.height))


class Goal:
    pos = None
    type = 0
    box = []

    def __init__(self, pos, type, box):
        self.pos = pos
        self.type = type
        self.box = box


class Agent:
    id = 0
    pos = None
    orientation = 0.0
    radius = 0.0
    goals = []
    color = []
    laser_history = []
    type_history = []
    laser_history_agents = []
    laser_history_obstacles = []
    laser_lines = []
    type_colors = []
    done = False

    # used in centralized env
    internal_states = None
    external_states_laser = None
    external_states_type = None

    def __init__(self, pos, radius, orientation, goals, id, color):
        self.pos = pos
        self.radius = radius
        self.orientation = orientation
        self.goals = goals
        self.id = id
        self.color = color
