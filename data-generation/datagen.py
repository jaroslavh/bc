from random import randint
import csv

#-------------------------------------------------------------------------
# Generates data in 3d space 0 to 1000 cube.
# Outputs a csv file in format cluster number, x, y, z
# Parameters:
#   cluster_size integer - number of points to generate in each cluster
#   cluster_number integer - number of clusters in space, capped at 8
#   file_name string - name of file to save the cluster
#   clear boolean - True - generate only clusters, False - generate some more random points
def generate_3d_data(cluster_size, cluster_number, file_name, clear):

    if (cluster_number) > 8:
        return False

    with open(file_name, 'w', newline='') as csv_file:
        data_writer = csv.writer(csv_file, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
        for cluster in range(0, cluster_number):
            x = 250 + 500 * ((cluster >> 2) % 2)
            y = 250 + 500 * ((cluster >> 1) % 2)
            z = 250 + 500 * ((cluster >> 0) % 2)
            for item in range(1, cluster_size):
                data_writer.writerow([cluster, randint(100, 200) + x, randint(100, 200)
                     + y, randint(100, 200) + z])
    if clear == False:
        generate_random_points(500, file_name)
    return True

#-------------------------------------------------------------------------
# Generates random points in 0 to 1000 space and appends them to csv file
# Format 100, x, y, z where 100 stands for no cluster but scattered random points
# Parameters:
#   number_of_points integer - how many points to generate
#   file_name string - to what file to append
def generate_random_points(number_of_points, file_name):
    with open(file_name, 'a', newline='') as csv_file:
        data_writer = csv.writer(csv_file, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
        for i in range(0, number_of_points):
            data_writer.writerow([100, randint(0, 1000), randint(0, 1000), randint(0, 1000)])

#-------------------------------------------------------------------------

print("Data generation started:")

if generate_3d_data( 200, 4, 'data.csv', True):
    print("Data generated.")
else:
    print("Data generation failed.")

