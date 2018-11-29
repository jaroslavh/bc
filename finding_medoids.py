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
        
        pointsList = [x for x in pointsList \
            if not (similarity(medoids[-1].coors, x.coors) < delta)]

    return Cluster(str(cluster.id), medoids)

# always adds to the medoids the farthest point
def greedyApproach(similarity, cluster, delta):
    pointsList = cluster.points
    farthestPoint = pointsList[0]
    medoids = []
    while len(pointsList) != 0:
        maxDist = 0
        medoids.append(farthestPoint)
        pointsList.remove(farthestPoint)
        pointsList = [x for x in pointsList \
            if not (similarity(medoids[-1].coors, x.coors) < delta)]

        for remaining in pointsList:
            tempDist = similarity(medoids[-1].coors, remaining.coors)
            if (tempDist > maxDist):
                maxDist = tempDist
                farthestPoint = remaining
    return Cluster(str(cluster.id), medoids)

# always adds the point that is closest to the middistance between last added
# point and farthest point
def smartGreedyApproach(similarity, cluster, delta):
    pointsList = cluster.points
    smartPoint = pointsList[0]
    medoids = []
    while pointsList:
        maxDist = 0
        medoids.append(smartPoint)

        # filtering out the points already in delta + finding farthest point
        pointsList = [x for x in pointsList \
            if not (similarity(medoids[-1].coors, x.coors) < delta)]

        for point in pointsList:
            tempDist = similarity(medoids[-1].coors, point.coors)
            if tempDist > maxDist:
                maxDist = tempDist
        
        optimalDistance = maxDist - delta
        currentDist = maxDist

        # finding the optimal medoid
        for remaining in pointsList:
            tempDist = similarity(medoids[-1].coors, remaining.coors)
            if (abs(optimalDistance - tempDist) <= currentDist):
                currentDist = abs(optimalDistance - tempDist)
                smartPoint = remaining

    return Cluster(str(cluster.id), medoids)

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