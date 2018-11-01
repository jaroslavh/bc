# implementation of class point
from scipy.spatial import distance

class Point:
    visited = False
    x = 0
    y = 0
    z = 0

    def __init__(self, x, y, z):
        self.x = x
        self.y = y
        self.z = z
        self.visited = False

    def coordinates(self):
        return [self.x, self.y, self.z]    