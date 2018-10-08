import matplotlib.pyplot as plt
import csv

with open('../datasets/3d_data_4_clusters.csv', newline='') as csv_file:
    data_file_reader = csv.reader(csv_file, delimiter=' ', quotechar='|')
    data_list = []
    for row in data_file_reader:
        data_list.append(row)

plt.plot(data_file_reader)
plt.ylabel('some numbers')
plt.show()
