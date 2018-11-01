#implementation of algorithms
import csv
import sys
from scipy.spatial import distance

from point import Point
from cluster import Cluster

#read the data file
in_file = sys.argv[1] # TODO work with CLI arguments
delta = sys.argv[2]
points_from_file = []

with open(in_file, newline='') as csv_file:
    data_file_reader = csv.reader(csv_file, delimiter=' ', quotechar='|')

    for row in data_file_reader:
        split = row[0].split(',')
        points_from_file.append(Point(float(split[1]), float(split[2]), float(split[3])))

#whole file loaded to memory we can start to go through the points
cluster = Cluster(0, points_from_file[0])
del[points_from_file[0]]
while points_from_file:
    for point in points_from_file:
        if (distance.cosine(cluster.medoids[-1], point) < delta):
            # append list
            # delete it from points_from file
