#implementation of class cluster
from point import Point

class Cluster:
    medoids = []
    id = 0

    def __init__(self, id, initial_medoid):
        self.medoids.append(initial_medoid)
        self.id = id