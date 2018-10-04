import json
from random import randint

#-------------------------------------------------------------------------
# filling table example for size 4
#  XX | a0 | a1 | a2 | a3
# ------------------------
#  a0 | XX |    |    |
# ------------------------
#  a1 |fill| XX |    |
# ------------------------
#  a2 |fill|fill| XX |
# ------------------------
#  a3 |fill|fill|fill| XX
# ------------------------
# from this table if the bracket fill is higher than 800, then I create a subtable with the
# row and column
# returns { id : {id_of_peer : num_of_hits }}
def generate_similarity_tables(size):
    
    ret_dict = {}
    id_num = 0
    while id_num != size: # generate identificators field
        ret_dict.update({"a" + str(id_num) : []})
        id_num += 1

    for row in ret_dict:
        for column in ret_dict:
            if(row == column):
                continue
            app = randint(0,1000)
            if (app > 800): # set threshold for data to be taken into account
                ret_dict[row].append({column:app})
                ret_dict[column].append({row:app})
    return ret_dict

#-------------------------------------------------------------------------

print("Starting to generate data")

data = generate_similarity_tables(1000)


output_file = open("data.json", 'w')

output_file.write(json.dumps(data))
