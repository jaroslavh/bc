import numpy as np
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt
import csv
import sys
from scipy.spatial import distance

# checking arguments
if len(sys.argv) != 3:
    print("Usage: python3 similarity_histogram.py <input filename> <cosine/euclidean dist>")
    exit(1)

in_file = sys.argv[1]
if (sys.argv[2] == '-e'):
    print("Calculating histogram for euclidean distance.")
elif (sys.argv[2] == '-c'):
    print("Calculating histogram for cosine similarity.")
else: 
    print("Distance argument error. Use: -e - euclidean, -c - cosine")
    exit(1)

distance_measure = sys.argv[2] # euclidean distance of cosine similarity
histogram_data = [] # list to load data from csv
 
with open(in_file, newline='') as csv_file:
    data_file_reader = csv.reader(csv_file, delimiter=' ', quotechar='|')

    reference_point = next(data_file_reader)[0].split(',')
    base_point = [float(reference_point[1]), float(reference_point[2]), float(reference_point[3])]

    for row in data_file_reader:
        split = row[0].split(',')
        distant_point =  [float(split[1]), float(split[2]), float(split[3])]
        if (distance_measure == '-e'):
            histogram_data.append(distance.euclidean(base_point, distant_point))
        else:    
            histogram_data.append(distance.cosine(base_point, distant_point))

for i in [30]: #[10, 20, 30, 40, 50]:
    num_bins = i
    patches = plt.hist(histogram_data, num_bins, facecolor='blue', alpha=0.5)
    plt.show()
