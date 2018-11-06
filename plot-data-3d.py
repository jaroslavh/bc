# script to plot clusters loaded from csv file in format:
# cluster_name, x_float, y_float, z_float
# 
# script does not check the structure of the file - expects right format

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import csv
import sys

if len(sys.argv) != 2:
    print("Usage: python3 plot-data-3d.py <input filename>")
    exit(1)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

dataset_folder = "datasets/"
x_data = []
y_data = []
z_data = []

colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'w', 'b']
colors_counter = 0

in_file = sys.argv[1]

with open(in_file, newline='') as csv_file:
    data_file_reader = csv.reader(csv_file, delimiter=' ', quotechar='|')
    cluster_number = None
    for row in data_file_reader:
        split = row[0].split(',')

        if (cluster_number == None):
            cluster_number = split[0]
        elif (cluster_number != split[0]):
            ax.scatter(x_data, y_data, z_data, c=colors[colors_counter], marker='o')
            cluster_number = split[0]
            colors_counter += 1
            x_data = []
            y_data = []
            z_data = []
            
        x_data.append(float(split[1]))
        y_data.append(float(split[2]))
        z_data.append(float(split[3]))   

# plot the last cluster
ax.scatter(x_data, y_data, z_data, c=colors[colors_counter], marker='o')

xLabel = ax.set_xlabel('X')
yLabel = ax.set_ylabel('Y')
zLabel = ax.set_zlabel('Z')

# Uncomment if you want to plot 1000x1000x1000 graphs
#ax.set_xlim([0, 1000])
#ax.set_ylim([0, 1000])
#ax.set_zlim([0, 1000])

plt.show()
