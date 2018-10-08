from random import randint
import csv

#-------------------------------------------------------------------------
def generate_3d_data(cluster_size, cluster_number, file_name):

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
    return True

#-------------------------------------------------------------------------

print("Data generation started:")

if generate_3d_data( 200, 4, 'data.csv'):
    print("Data generated.")
else:
    print("Data generation failed.")

