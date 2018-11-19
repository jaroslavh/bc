#implementation of class cluster
import csv
from point import Point

class Cluster:
    points = [] # points in cluster
    id = None # identification of cluster

    def __init__(self, id, inPoints=None):
        if inPoints:
            self.points = inPoints
        self.id = id

    def __str__(self): #TODO what about ID?
        stringList = []
        for point in self.points:
            stringList.append(str(point))
        return str(stringList)

    def __eq__(self, other):
        if (self.id != other.id):
            return False

        if (len(self.points) != len(other.points)):
            return False

        i = 0 
        while i < len(self.points):
            if (self.points[i] != other.points[i]):
                return False
            i += 1
        return True        

    def append(self, point):
        self.points.append(point)

    def writeFile(self, out_file):
        with open(out_file, 'a', newline='') as csv_file:
            data_writer = csv.writer(csv_file, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
            for point in self.points:
                data_writer.writerow([id] + point.coors)