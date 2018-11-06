#implementation of class cluster
import csv
from point import Point

class Cluster:
    medoids = []
    id = 0

    def __init__(self, id):
        self.id = id

    def append(self, point):
        self.medoids.append(point)

    def writeFile(self, out_file):
        with open(out_file, 'a', newline='') as csv_file:
            data_writer = csv.writer(csv_file, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
            for point in self.medoids:
                data_writer.writerow([self.id, point.x, point.y, point.z])