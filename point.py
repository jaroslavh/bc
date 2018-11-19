# implementation of class point

class Point:
    coors = [] # coordinates of point

    def __init__(self, coordinates): #TODO check if it is a list of ints
        self.coors = coordinates

    def __str__(self):
        return str(self.coors)

    def __eq__(self, other):
        return (self.coors == other.coors)

    def dim(self):
        return len(self.coors)