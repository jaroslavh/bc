# Reading and writing to files with Cluster representation

import csv
import os.path

from cluster import Cluster
from point import Point

# Class that represents one file.
#   path - string with path to the file
#   fileAction - in what mode do we open the file (read/write/append)
class FileIO:
    path = None
    csvFile = None

    def __init__(self, filePath, fileAction):
        if ( (fileAction == "r") or (fileAction == "a") ):
            try:
                self.csvFile = open(filePath, fileAction, newline='')
            except Exception as e:
                raise(e)
        elif ( fileAction == "w" ):
            if ( os.path.isfile(filePath) ):
                raise(FileExistsError)

        self.path = filePath
    
    def close(self):
        self.csvFile.close()


    # returns a list of clusters
    def loadFile(self):
        dataFileReader = csv.reader(self.csvFile, delimiter=' ', quotechar='|')
        clusterID = None
        retClusters = []
        clusterPoints = []

        for row in dataFileReader:
            split = row[0].split(',')

            if (clusterID != split[0]):
                if(clusterID != None):
                    retClusters.append(Cluster(clusterID, clusterPoints))
                    clusterPoints = []

                clusterID = split[0]

            pointList = list(map(float, split[1:]))
            clusterPoints.append(Point(pointList))

        retClusters.append(Cluster(clusterID, clusterPoints))
        return retClusters
