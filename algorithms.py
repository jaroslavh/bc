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
    farthestPoint = pointsList[0]
    pointsList.remove(farthestPoint)
    while pointsList:
        maxDist = 0
        cluster.append(farthestPoint) #starting with a random point
        for point in pointsList:
            tempDist = similarity(cluster.medoids[-1].coordinates(), point.coordinates())
            if (tempDist > maxDist):
                maxDist = tempDist
                farthestPoint = point
            pointsList.remove(point)

def smartGreedyApproach(similarity, pointsList, cluster, delta):
    smartPoint = pointsList[0]
    while pointsList:
        cluster.append(smartPoint) #starting with a random point
        pointsList.remove(smartPoint)
        maxDist = 0

        for point in pointsList:
            tempDist = similarity(smartPoint.coordinates(), point.coordinates())
            if (tempDist <= delta):
                pointsList.remove(point)
            if (tempDist > maxDist):
                maxDist = tempDist

        optimalDistance = delta + ((maxDist - delta) / 2)
        currentDist = optimalDistance

        for remaining in pointsList:
            tempDist = similarity(smartPoint.coordinates(), remaining.coordinates())
            if (abs(optimalDistance - tempDist) < currentDist):
                currentDist = abs(optimalDistance - tempDist)
                smartPoint = remaining
        

# check cli arguments
if len(sys.argv) != 5:
    print("Usage: python3 algorithms.py <input filename> <output filename> <cosine/euclidean dist> <delta list>")
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
delta_cnt = 0
delta_list = [350, 350, 160]
points_from_file = []

# read the data file

with open(in_file, newline='') as csv_file:
    data_file_reader = csv.reader(csv_file, delimiter=' ', quotechar='|')
    cluster_number = None
    for row in data_file_reader:
        split = row[0].split(',')

        if (cluster_number == None):
            cluster_number = split[0]
            cluster = Cluster(cluster_number)
        elif (cluster_number != split[0]):
            cluster_number = split[0]

            # whole file loaded to memory we can start to go through the points
            #randomPoints(similarity_func, points_from_file, cluster, delta_list[delta_cnt])
            greedyApproach(similarity_func, points_from_file, cluster, delta_list[delta_cnt])
            #smartGreedyApproach(similarity_func, points_from_file, cluster, delta_list[delta_cnt])
            cluster.writeFile(out_file)
            cluster = Cluster(cluster_number)
            cluster.medoids = []
            delta_cnt += 1
            points_from_file = []
            
        points_from_file.append(Point(float(split[1]), float(split[2]), float(split[3])))
        
# whole file loaded to memory we can start to go through the points
#randomPoints(similarity_func, points_from_file, cluster, delta_list[delta_cnt])
greedyApproach(similarity_func, points_from_file, cluster, delta_list[delta_cnt])
#smartGreedyApproach(similarity_func, points_from_file, cluster, delta_list[delta_cnt])
cluster.writeFile(out_file)