import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import csv
import sys

if len(sys.argv) != 2:
    print("Usage: python3 plot-data-3d.py <input filename>")
    exit(1)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

x_data = []
y_data = []
z_data = []

with open(sys.argv[1], newline='') as csv_file:
    data_file_reader = csv.reader(csv_file, delimiter=' ', quotechar='|')
    for row in data_file_reader:
        split = row[0].split(',')
        x_data.append(float(split[1]))
        y_data.append(float(split[2]))
        z_data.append(float(split[3]))
   
ax.scatter(x_data, y_data, z_data, c='r', marker='o')
xLabel = ax.set_xlabel('X')
yLabel = ax.set_ylabel('Y')
zLabel = ax.set_zlabel('Z')
ax.set_xlim([0, 1000])
ax.set_ylim([0, 1000])
ax.set_zlim([0, 1000])

plt.show()
