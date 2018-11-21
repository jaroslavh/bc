#!/usr/bin/env python3

#implementation of algorithms
import csv
import sys
from scipy.spatial import distance
import argparse

from point import Point
from cluster import Cluster
from fileio import FileIO

# returns cluster with only medoids present
def randomPoints(similarity, cluster, delta):
    pointsList = cluster.points
    medoids = []
    while pointsList:
        medoids.append(pointsList.pop())
        for point in pointsList:
            if (similarity(medoids[-1].coors, point.coors) < delta):
                pointsList.remove(point)
    return Cluster("med" + str(cluster.id), medoids)

# always adds to the medoids the farthest point
def greedyApproach(similarity, cluster, delta):
    pointsList = cluster.points
    farthestPoint = pointsList[0]
    medoids = []
    while pointsList:
        maxDist = 0
        medoids.append(farthestPoint)
        pointsList.remove(farthestPoint)
        for point in pointsList:
            if (similarity(farthestPoint.coors, point.coors) <= delta):
                pointsList.remove(point)

        for point in pointsList:
            tempDist = similarity(farthestPoint.coors, point.coors)
            if (tempDist > maxDist):
                maxDist = tempDist
                farthestPoint = point
    return Cluster("med" + str(cluster.id), medoids)

# always adds the point that is closest to the middistance between last added
# point and farthest point
def smartGreedyApproach(similarity, cluster, delta):
    pointsList = cluster.points
    smartPoint = pointsList[0]
    medoids = []
    while pointsList:
        maxDist = 0
        medoids.append(smartPoint)
        pointsList.remove(smartPoint)

        # filtering out the points already in delta + finding farthest point
        for point in pointsList:
            tempDist = similarity(medoids[-1].coors, point.coors)
            if (tempDist <= delta):
                pointsList.remove(point)
            elif (tempDist > maxDist):
                maxDist = tempDist
        
        optimalDistance = delta + ((maxDist - delta) / 2)
        currentDist = optimalDistance

        # finding the optimal medoid
        for remaining in pointsList:
            tempDist = similarity(smartPoint.coors, remaining.coors)

            if (abs(optimalDistance - tempDist) < currentDist):
                currentDist = abs(optimalDistance - tempDist)
                smartPoint = remaining

    return Cluster("med" + str(cluster.id), medoids)

# Based on [neighbor_number] returns to which cluster the [point]
# belongs from [representation_file].       
def kNearestNeighbors(point, representation_file, neighbor_number):
    print("Nearest Neighbors")


# check cli arguments
parser = argparse.ArgumentParser()
parser.add_argument("inFile", help="csv file to read data from")
parser.add_argument("outFile", help="csv file to store data in")
parser.add_argument("similarityMeasure", help="similarity masure \
    [e (euclidean), c (cosine)")
args = parser.parse_args()
in_file = args.inFile
out_file = args.outFile
if (args.similarityMeasure == 'e'):
    similarity_func = distance.euclidean
elif (args.similarityMeasure == 'c'):
    similarity_func = distance.cosine
else: #TODO can this be done by argparse?
    print("Unknown similarity measure.")
    exit(1)
# TODO add argument for delta_list that I calculated from the histogram

delta_list = [350, 350, 160]

# read the data file
f = FileIO(in_file, "r")
clusters = f.loadFile()
f.close()

retClusters = []
deltaCnt = 0
for cluster in clusters:
    retClusters.append(
        smartGreedyApproach(similarity_func, cluster, delta_list[deltaCnt]))
    deltaCnt += 1
        
for cluster in retClusters:
    cluster.writeFile(out_file)