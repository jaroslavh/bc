#implementation of algorithms
import csv
import sys
from scipy.spatial import distance

from point import Point
from cluster import Cluster

def randomPoints(similarity, pointsList, cluster, delta):
    while pointsList:
        cluster.append(pointsList[0]) #starting with a random point
        del(pointsList[0])
        for point in pointsList:
            if (similarity(cluster.medoids[-1].coordinates(), point.coordinates()) < delta):
                pointsList.remove(point)

def greedyApproach(similarity, pointsList, cluster, delta):
    farhtestPoint = pointsList[0]
    while pointsList:
        cluster.append(farhtestPoint) #starting with a random point
        pointsList.remove(farhtestPoint)
        maxDist = 0
        for point in pointsList:
            tempDist = similarity(cluster.medoids[-1].coordinates(), point.coordinates())
            if (tempDist < delta):
                pointsList.remove(point)
                continue
            if (tempDist > maxDist):
                maxDist = tempDist
                farhtestPoint = point

def smartGreedyApproach(similarity, pointsList, cluster, delta): #TODO check this algorithm
    smartPoint = pointsList[0]
    while pointsList:
        cluster.append(smartPoint) #starting with a random point
        pointsList.remove(smartPoint)
        maxDist = 0
        workList = []
        for point in pointsList:
            tempDist = similarity(cluster.medoids[-1].coordinates(), point.coordinates())
            if (tempDist < delta):
                pointsList.remove(point)
            else:
                workList.append((point, tempDist))
                if (tempDist > maxDist):
                    maxDist = tempDist
        
        optimalDistance = delta + ((maxDist - delta) / 2)
        currentDist = optimalDistance

        for remaining in workList:
            if (optimalDistance - remaining[1] < currentDist):
                currentDist = optimalDistance - remaining[1]
                smartPoint = remaining[0]
        

# check cli arguments
if len(sys.argv) != 5:
    print("Usage: python3 algorithms.py <input filename> <output filename> <cosine/euclidean dist> <delta>")
    exit(1)

in_file = sys.argv[1]
out_file = sys.argv[2]
similarity_measure = sys.argv[3]
if (similarity_measure == '-e'):
    print("Calculating histogram for euclidean distance.")
    similarity_func = distance.euclidean
elif (similarity_measure == '-c'):
    print("Calculating histogram for cosine similarity.")
    similarity_func = distance.cosine
else: 
    print("Distance argument error. Use: -e - euclidean, -c - cosine")
    exit(1)
delta = float(sys.argv[4])
points_from_file = []

# read the data file
with open(in_file, newline='') as csv_file:
    data_file_reader = csv.reader(csv_file, delimiter=' ', quotechar='|')

    for row in data_file_reader:
        split = row[0].split(',')
        points_from_file.append(Point(float(split[1]), float(split[2]), float(split[3])))

# whole file loaded to memory we can start to go through the points
cluster = Cluster(0)

#randomPoints(similarity_func, points_from_file, cluster, delta)
#greedyApproach(similarity_func, points_from_file, cluster, delta)
smartGreedyApproach(similarity_func, points_from_file, cluster, delta)
        
cluster.writeFile(out_file)