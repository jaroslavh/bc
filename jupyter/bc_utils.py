import pandas as pd
import numpy as np
import random

from scipy.spatial import distance
from matplotlib import pyplot


# utils for usage in multiple other notebooks

class TestDf:
    train_df = None
    test_df = None
    full_df = None
    split = None
    state = 0

    def __init__(self, init_df):
        self.state = 0
        self.split = [int(len(init_df) * 0.2),
                 int(len(init_df) * 0.4),
                 int(len(init_df) * 0.6),
                 int(len(init_df) * 0.8)]
        self.full_df = init_df

        self.rotate()
    
    #rotate for new set in cross validation    
    def rotate(self):
        self.state = self.state + 1
        if (self.state == 5):
            self.state = 0

        # below are all the possible splits of full dataset
        #   T stands for 20% of dataset used for testing
        #   X stands for the rest 80% of dataset used for training
        # TXXXX
        if (self.state == 0):
            self.test_df = self.full_df.iloc[:self.split[0]]
            self.train_df = self.full_df.iloc[self.split[0]:]
        # XTXXX
        if (self.state == 1): 
            self.test_df = self.full_df.iloc[self.split[0]:self.split[1]]
            self.train_df = self.full_df.iloc[:self.split[0]].append(self.full_df.iloc[self.split[1]:])
        # XXTXX
        if (self.state == 2): 
            self.test_df = self.full_df.iloc[self.split[1]:self.split[2]]
            self.train_df = self.full_df.iloc[:self.split[1]].append(self.full_df.iloc[self.split[2]:])
        # XXXTX
        if (self.state == 3): 
            self.test_df = self.full_df.iloc[self.split[2]:self.split[3]]
            self.train_df = self.full_df.iloc[:self.split[2]].append(self.full_df.iloc[self.split[3]:])
        # XXXXT
        if (self.state == 4): 
            self.test_df = self.full_df.iloc[self.split[3]:]
            self.train_df = self.full_df.iloc[:self.split[3]]

    # returns the internal state of current split
    def get_state():
        return self.state

    #shows plot of given dataframe
    def show_plot(self):
        fix, ax = pyplot.subplots()

        # adding traning data to plot
        train_colors = ['#FF99FF', '#A6A6FF']
        grouped = self.train_df.groupby('label')
        for key, group in grouped:
            group.plot(ax=ax, kind='scatter', x='x', y='y',
                   label="train_"+str(key), color=train_colors[key])
        # adding testing data to plot
        test_colors = ['#FF0000', '#0000FF']
        grouped = self.test_df.groupby('label')
        for key, group in grouped:
            group.plot(ax=ax, kind='scatter', x='x', y='y',
                   label="test_"+str(key), color=test_colors[key])
        pyplot.show()

#def delta_medoids_one_shot(df, delta, similarity_measure):
#    """Returns subset of input DataFrame, that is a good representation
#    of given data.
#
#    This is a simplified delta-medoids algorithm. It finds out the
#    representatives in one pass through the input data. Final representatives
#    depend on the ordering of input data.
#
#    :param df: in data
#    :type df: pandas.DataFrame
#    :param delta: maximum distance of points to be considered similar
#    :type delta: float
#    :param similarity_measure: similarity function to be used in algorithm
#    :type similarity_measure: scipy.spacial.distance
#
#    :Example:
#    >>> TODO"""
#    representatives = np.array(df.iloc[0], ndmin=2)
#
#    #here starts RepAssign routine for advanced delta-medoids
#    for row in df.iterrows():
#        point = tuple(row[1])
#
#        for rep in representatives: #needs optimalization
#            if similarity_measure(point, rep) <= delta:
#                break
#        else:
#            representatives = np.vstack((representatives, point))
#
#    return pd.DataFrame(representatives, columns=df.columns.values)

def delta_medoids_one_shot(df, delta, similarity_measure):
    """Returns subset of input DataFrame, that is a good representation
    of given data.

    This is a simplified delta-medoids algorithm. It finds out the
    representatives in one pass through the input data. Final representatives
    depend on the ordering of input data.

    :param df: in data
    :type df: pandas.DataFrame
    :param delta: maximum distance of points to be considered similar
    :type delta: float
    :param similarity_measure: similarity function to be used in algorithm
    :type similarity_measure: scipy.spacial.distance

    :Example:
    >>> TODO"""
    representatives = np.array(df.iloc[0], ndmin=2)

    for row in df.iterrows():
        dist = float("inf")
        point = tuple(row[1])

        for rep in representatives:
            #finding the closest representative to current point
            if similarity_measure(point, rep) <= dist:
                dist = similarity_measure(point, rep)
        if dist > delta:
            representatives = np.vstack((representatives, point))

    return pd.DataFrame(representatives, columns=df.columns.values)

# this is the argmin calculation for delta_medoids_full algorithm
def find_best_cluster_representative(cluster, similarity_measure):
    #input: np.array(of points)
    #returns a tuple

    min_sum = float("inf")
    best_repr_index = None
    for i in range(len(cluster)):
        distance_sum = 0;
        for point in cluster:
            distance_sum += similarity_measure(cluster[i], point)
            if (distance_sum > min_sum):
                break
        if(distance_sum < min_sum):
            min_sum = distance_sum
            best_repr_index = i

    return cluster[best_repr_index]

def delta_medoids_full(df, delta, similarity_measure):

    t = 0 #iteration number
    representatives = {} #selected representatives
    representatives[t] = set() #representatives for given iteration
    size_threshold = 0.01 * len(df.index)

    while True:
        #print("\n=========== running for t = " + str(t) + "============")
        clusters = {} #subclusters inside cluster
        for item in representatives[t]:
            clusters[item] = np.array(item, ndmin=2)

        t = t + 1

        #================== RepAssign starts ==================
        for row in df.iterrows():
            dist = float("inf")
            represenative = None

            point = tuple(row[1])

            for rep in clusters.keys():
                #finding the closest representative to current point
                if similarity_measure(point, rep) <= dist:
                    representative = rep
                    dist = similarity_measure(point, rep)
            if dist <= delta:
                clusters[representative] = np.vstack((clusters[representative], point))
            elif t == 1:
                clusters[point] = np.array(point, ndmin=2)
        #================== RepAssign ends ===================

        
        if (len(clusters.keys()) > 50) or (len(clusters.keys()) > len(df.index) * 0.1):
            for key in clusters.keys():
                if len(clusters[key]) <= size_threshold:
                    tmp_sim = 0
                    max_sim = 0
                    new_key = key
                    for key_sim in clusters.keys():
                        tmp_sim = similarity_measure(key, key_sim)
                        if key_sim == key:
                            continue
                        elif tmp_sim > max_sim:
                            new_key = key_sim
                            max_sim = tmp_sim
                
#                     if new_key == key:
#                         print('Dropping this cluster, it is not relevant.')
                    if new_key != key: #else:
                        clusters[new_key] = np.vstack((clusters[new_key], clusters[key]))
                    del(clusters[key])
#                     print 'clusters remaining', len(clusters.keys())
        
        representatives[t] = set()
        for cluster in clusters.values():
            representative = find_best_cluster_representative(cluster, similarity_measure)
            #print(representative)
            representatives[t].add(tuple(representative))
        #print(representatives[t])

        if representatives[t] == representatives[t-1]:
            break

    #print("delta_medoids_full algorithm ended after " + str(t) + " iterations.")
    return pd.DataFrame(list(representatives[t]), columns=df.columns.values)

def delta_medoids_full_old(df, delta, similarity_measure):
    """Returns subset of input DataFrame, that is a good representation
    of given data.

    This is a simplified delta-medoids algorithm. It finds out the
    representatives in one pass through the input data. Final representatives
    depend on the ordering of input data.

    :param df: in data
    :type df: pandas.DataFrame
    :param delta: maximum distance of points to be considered similar
    :type delta: float
    :param similarity_measure: similarity function to be used in algorithm
    :type similarity_measure: scipy.spacial.distance

    :Example:
    >>> TODO"""
    t = 0 #iteration number
    representatives = {} #selected representatives
    representatives[t] = set() #representatives for given iteration

    while True:
        #print("\n=========== running for t = " + str(t) + "============")
        clusters = {} #subclusters inside cluster
        for item in representatives[t]:
            clusters[item] = np.array(item, ndmin=2)

        t = t + 1

        #================== RepAssign starts ==================
        for row in df.iterrows():
            dist = float("inf")
            represenative = None

            point = tuple(row[1])

            for rep in clusters.keys():
                #finding the closest representative to current point
                if similarity_measure(point, rep) <= dist:
                    representative = rep
                    dist = similarity_measure(point, rep)
            if dist <= delta:
                clusters[representative] = np.vstack((clusters[representative], point))
            else:
                clusters[point] = np.array(point, ndmin=2)
        #================== RepAssign ends ===================

        representatives[t] = set()
        for cluster in clusters.values():
            representative = find_best_cluster_representative(cluster, similarity_measure)
            #print(representative)
            representatives[t].add(tuple(representative))
        #print(representatives[t])

        if representatives[t] == representatives[t-1]:
            break

    #print("delta_medoids_full algorithm ended after " + str(t) + " iterations.")
    return pd.DataFrame(list(representatives[t]), columns=df.columns.values)

def estimate_delta(df, similarity_measure):
    """Estimate delta for given dataset and similarity measure.

    Uses heuristic by picking 3 points from given dataframe - first, last and
    middle one. Then it calculates distance to each other point in the dataset
    and stores them. It returns the median * 1.05 value as a estimate delta value
    for given dataset"""
    similarities = np.array([])
    
    ref_points = [df.iloc[0].values,
            df.iloc[len(df.index) / 2].values,
            df.iloc[len(df.index) - 1].values]

    for ref_point in ref_points:
        for row in df.iterrows():
            point = tuple(row[1])
            
            #skipping current ref_point
            if point == tuple(ref_point):
                continue

            sim = similarity_measure(point, ref_point)
            similarities = np.append(similarities, sim)

    # aiming to get 10 representatives for the whole dataset, though picking from sorted value of similarities
    # this counts on uniform distribution of values
    delta = np.sort(similarities)[similarities.size / 10]
    return delta * 1.05 #TODO store this value somewhere better

def random_select(df, num):
    """Returns random subset from input DataFrame of given size.

    :param df: in data
    :type df: pandas.DataFrame
    :param num: number of representatives to select
    :type num: int

    :Example:
    >>> ret_df = random_select(input_df, 5)"""
    rows = random.sample(df.index, num)
    representatives = df.loc[rows]
    return representatives

# finding representation of a cluster with random selection
# pandas.DataFrame in_df
# float delta
def greedy_select(df, delta, similarity_measure):
    """Returns subset from input DataFrame selected by greedy algorithm that
    that always looks for representative that if the farthest from all
    previously selected.

    All others are calculate by finding the smallest sum of similarities to all
    representatives previously selected.

    :param df: in data
    :type df: pandas.DataFrame
    :param delta: maximum distance of points to be considered similar
    :type delta: float
    :param similarity_measure: similarity function to be used in algorithm
    :type similarity_measure: scipy.spacial.distance

    :Example:
    >>> ret_df = greedy_select(input_df, 0.25, distance.cosine)"""
    
    medoid_indexes = np.array([])
    index_list = np.array(df.index.values)
    
    while (index_list.size > 0):
        
        # selecting the first index
        if not (medoid_indexes.size > 0):
            current_index = df.loc[index_list[0]].name
        # finding index of the most different point
        else:
            for index in index_list: #TODO does this copy the list
                current_index = None
                similarity_sum = 0 # used to store the biggest sum
                tmp_sum = 0 # used to store the current sum
                for medoid in medoid_indexes:
                    tmp_sum = tmp_sum + similarity_measure(df.loc[index].values[:],
                                                           df.loc[medoid].values[:])
                if tmp_sum > similarity_sum:
                    similarity_sum = tmp_sum
                    current_index = index
    
        rem_indexes = np.array([])
        
        # find indexes to drop from dataframe
        for ix in df.index:
            if similarity_measure(df.loc[ix].values[:], df.loc[current_index].values[:]) <= delta:
                rem_indexes = np.append(rem_indexes, ix)
        
        # adding selected index to other representatives
        medoid_indexes = np.append(medoid_indexes, current_index)
        
        # dropping points from the delta distance of newly selected representative
        index_list = np.array([item for item in index_list if item not in rem_indexes])
        rem_indexes = np.delete(rem_indexes, np.where(rem_indexes == current_index))
        df = df.drop(rem_indexes)

    return df

def greedy_select_real(df, delta, similarity_measure):
    
    representatives = np.array([], str)                      # selected representatives
    remaining = np.array(df.index.values, str)               # remaining indexes not covered
    
    while (remaining.size > 0):
        if (representatives.size == 0):                 # selecting the first index
            current_index = df.loc[remaining[0]].name
        else:                                           # finding index of the most different point
            current_index = None
            sim_sum = representatives.size              # used to store the biggest sum
            for index in remaining:
                tmp_sum = 0                             # used to store the current sum
                for medoid in representatives:          # finding different the point is
                    tmp_sum += similarity_measure(df.loc[index]['data'], df.loc[medoid]['data'])
                if tmp_sum < sim_sum:
                    sim_sum = tmp_sum
                    current_index = index
                    
        # adding selected index to other representatives
        representatives = np.append(representatives, current_index)
        
        # find indexes to drop from dataframe
        rem_indexes = [i for i in remaining if similarity_measure(df.loc[i]['data'], df.loc[current_index]['data']) >= delta]
        
        rem_indexes = np.array([], str)
        for ix in remaining:
            if similarity_measure(df.loc[ix]['data'], df.loc[current_index]['data']) >= delta:
                rem_indexes = np.append(rem_indexes, ix)
        
        # dropping points from the delta distance of newly selected representative
        remaining = np.array([item for item in remaining if item not in rem_indexes])
        #TODO DELETE? rem_indexes = np.delete(rem_indexes, np.where(rem_indexes == current_index))
        
    return df.loc[representatives]

def classifyPoints(ref_df, test_df):

    X_train = ref_df.iloc[:, :-1].values
    y_train = ref_df.iloc[:, -1].values
    X_test = test_df.iloc[:, :-1].values
    y_test = test_df.iloc[:, -1].values

    from sklearn.preprocessing import StandardScaler  
    scaler = StandardScaler()  
    scaler.fit(X_train)

    X_train = scaler.transform(X_train)  
    X_test = scaler.transform(X_test)

    from sklearn.neighbors import KNeighborsClassifier  
    classifier = KNeighborsClassifier(n_neighbors=1)
    classifier.fit(X_train, y_train)

    y_pred = classifier.predict(X_test)

    from sklearn.metrics import classification_report, confusion_matrix
    conf_mat = confusion_matrix(y_test, y_pred)
    #print(conf_mat)  
    print(classification_report(y_test, y_pred))
    return conf_mat

def classifyPoints_real(ref_df, test_df, similarity):

    names = set(ref_df['cluster'])
    
    belongs = {}
    for name in names:
        belongs[name] = {}
        for n in names:
            belongs[name][n] = 0
    
    suc = 0
    fai = 0
    miss = 0
    for row in test_df.iterrows():
        sim = 0
        cluster = 'NONE'
        for ref in ref_df.iterrows():
            tmp_sim = similarity(row[1]['data'], ref[1]['data'])
            if tmp_sim > sim:
                sim = tmp_sim
                cluster = ref[1]['cluster']
                if sim == 1:
                    break
        
        if(sim == 0):
            miss += 1
            continue
        else:    
            belongs[row[1]['cluster']][cluster] += 1
        
        if cluster == row[1]['cluster']:
            suc += 1
        else:
            fai += 1
            
    print('Missed: ' + str(miss))
    print('Succeeded: ' + str(suc))
    print('Failed: ' + str(fai))
    print('Ratio: ' + str(float(suc) / float(suc + fai)))
    
    names = sorted(names)
    matrix = [[0 for i in names] for i in names]

    for i in range(0, len(names)):
        for j in range(0, len(names)):
            matrix[i][j] = belongs[names[i]][names[j]]
 
    return np.array(matrix)

#input is a precision recall matrix and this methore calculates the ratio
#it is used for comparison of algorithms
def get_hit_miss_rate(matrix):
    hit = 0
    miss = 0

    i = 0
    while i < len(matrix):
        j = 0
        while j < len(matrix[i]):
            if (i == j):
                hit = hit + matrix[i][j]
            else:
                miss = miss + matrix[i][j]
            j = j + 1
        i = i + 1

    return float(miss)/float(hit)

def get_method_results_updated(full_test_df, full_train_df, similarity, delta):
    test_res = {"delta_medoids_full" : {},
                "delta_medoids_one_shot" : {},
                "random_select" : {},
                "greedy_select" : {}}

    #creating training DataFrames for comparing oneshot and full delta medoids algorithm
    train_delta_medoids_full = pd.DataFrame()
    train_delta_medoids_one_shot = pd.DataFrame()
    train_random_selection = pd.DataFrame()
#     train_greedy_select = pd.DataFrame()    

    for name in set(full_train_df['cluster']):
        delta_df = full_train_df[full_train_df['cluster'] == name].iloc[:, :-1]
        print('Running for :' + str(name))
        print(len(delta_df.index))

        if delta == None: #estimating delta if none is set
            delta = estimate_delta(delta_df, similarity)
    
        #delta medoids full
        print('Doing delta-medoids Full')
        medoids_full_result = delta_medoids_full(delta_df, delta, similarity)
        medoids_full_result['cluster'] = name #setting a cluster name for result
        test_res["delta_medoids_full"][name] = medoids_full_result
        train_delta_medoids_full = train_delta_medoids_full.append(medoids_full_result)
    
        print('Doing delta-medoids ONE SHOT')
        #delta medoids one shot
        one_shot_medoids_result = delta_medoids_one_shot(delta_df, delta, similarity)
        one_shot_medoids_result['cluster'] = name
        test_res["delta_medoids_one_shot"][name] = one_shot_medoids_result
        train_delta_medoids_one_shot = train_delta_medoids_one_shot.append(one_shot_medoids_result)
        
        print('Doing Random Select')
        #random select
        random_select_result = random_select(delta_df, medoids_full_result.shape[0])
        random_select_result['cluster'] = name
        test_res["random_select"][name] = random_select_result
        train_random_selection = train_random_selection.append(random_select_result)
    
        #greedy select
#         greedy_select_result = greedy_select(delta_df, delta, similarity)
#         greedy_select_result['cluster'] = name
#         test_res["greedy_select"][name] = greedy_select_result
#         train_greedy_select = train_greedy_select.append(greedy_select_result)    
    
    names = set(full_test_df['cluster'])
    full = list()
    greedy = []
    one_shot = []
    delta_med = []
    random = []
    for name in names:
        full.append(len(full_train_df[full_train_df['cluster'] == name].index))
        greedy.append(0)#len(train_greedy_select[train_greedy_select['cluster'] == name].index))
        one_shot.append(len(train_delta_medoids_one_shot[train_delta_medoids_one_shot['cluster'] == name].index))
        delta_med.append(len(train_delta_medoids_full[train_delta_medoids_full['cluster'] == name].index))
        random.append(len(train_random_selection[train_random_selection['cluster'] == name].index))
    

    titles = ['Cluster', 'Full', 'Greedy', 'One Shot', 'Delta Medoids', 'Random Select']
    data = [titles] + list(zip(names, full, greedy, one_shot, delta_med, random))
    
    for i, d in enumerate(data):
        line = '|'.join(str(x).ljust(12) for x in d)
        print(line)
        if i == 0:
            print('-' * len(line)) 
    
    print('Full Medoids')
    matrix_full = classifyPoints(train_delta_medoids_full, full_test_df)
    print('Delta Medoids')
    matrix_one_shot = classifyPoints(train_delta_medoids_one_shot, full_test_df)
    print('Random Selection')
    matrix_random_selection = classifyPoints(train_random_selection, full_test_df)
#     print('Greedy Selection')
#     matrix_greedy_select = classifyPoints(train_greedy_select, full_test_df)
    return [matrix_full, matrix_one_shot, matrix_random_selection] # matrix_greedy_select]


def get_method_results(full_test_df, dataframes, similarity, delta):
    test_res = {"delta_medoids_full" : {},
                "delta_medoids_one_shot" : {},
                "random_select" : {},
                "greedy_select" : {}}

    #creating training DataFrames for comparing oneshot and full delta medoids algorithm
    train_delta_medoids_full = pd.DataFrame()
    train_delta_medoids_one_shot = pd.DataFrame()
    train_random_selection = pd.DataFrame()
    train_greedy_select = pd.DataFrame()    

    for name in set(full_test_df['cluster']):
        delta_df = dataframes[name].train_df.iloc[:, :-1]

        if delta == None: #estimating delta if none is set
            delta = estimate_delta(delta_df, similarity)
    
        #delta medoids full
        medoids_full_result = delta_medoids_full(delta_df, delta, similarity)
        medoids_full_result['cluster'] = name #setting a cluster name for result
        test_res["delta_medoids_full"][name] = medoids_full_result
        train_delta_medoids_full = train_delta_medoids_full.append(medoids_full_result)
    
        #delta medoids one shot
        one_shot_medoids_result = delta_medoids_one_shot(delta_df, delta, similarity)
        one_shot_medoids_result['cluster'] = name
        test_res["delta_medoids_one_shot"][name] = one_shot_medoids_result
        train_delta_medoids_one_shot = train_delta_medoids_one_shot.append(one_shot_medoids_result)
    
        #random select
        random_select_result = random_select(delta_df, medoids_full_result.shape[0])
        random_select_result['cluster'] = name
        test_res["random_select"][name] = random_select_result
        train_random_selection = train_random_selection.append(random_select_result)
    
        #greedy select
        greedy_select_result = greedy_select(delta_df, delta, similarity)
        greedy_select_result['cluster'] = name
        test_res["greedy_select"][name] = greedy_select_result
        train_greedy_select = train_greedy_select.append(greedy_select_result)    
    
    names = set(full_test_df['cluster'])
    full = list()
    greedy = []
    one_shot = []
    delta_med = []
    random = []
    for name in names:
        full.append(4 * len(full_test_df[full_test_df['cluster'] == name].index))
        greedy.append(len(train_greedy_select[train_greedy_select['cluster'] == name].index))
        one_shot.append(len(train_delta_medoids_one_shot[train_delta_medoids_one_shot['cluster'] == name].index))
        delta_med.append(len(train_delta_medoids_full[train_delta_medoids_full['cluster'] == name].index))
        random.append(len(train_random_selection[train_random_selection['cluster'] == name].index))
    

    titles = ['Cluster', 'Full', 'Greedy', 'One Shot', 'Delta Medoids', 'Random Select']
    data = [titles] + list(zip(names, full, greedy, one_shot, delta_med, random))
    
    for i, d in enumerate(data):
        line = '|'.join(str(x).ljust(12) for x in d)
        print(line)
        if i == 0:
            print('-' * len(line)) 
    
    print('Full Medoids')
    matrix_full = classifyPoints(train_delta_medoids_full, full_test_df)
    print('Delta Medoids')
    matrix_one_shot = classifyPoints(train_delta_medoids_one_shot, full_test_df)
    print('Random Selection')
    matrix_random_selection = classifyPoints(train_random_selection, full_test_df)
    print('Greedy Selection')
    matrix_greedy_select = classifyPoints(train_greedy_select, full_test_df)
    return [matrix_full, matrix_one_shot, matrix_random_selection, matrix_greedy_select]

#TODO return also sizes of selections to evaluate how many of the dataset percentagewise it is

#methods for real data

#this method will take one 5min timewindow file
# and update the data structure based on it
def read_batch_file(path, data):
    data_file = pd.read_csv(path, sep='\t')
    #structure to load information from file into 
    # {user1:{host1:freq, host2:freq}, user2:{host1:freq, host2:freq}}
    if data == None:
        data = {}
        
    for index, row in data_file.iterrows():
        #extract information from hostnamePort
        raw_hostnames = row['hostnamePort'].split(';')    
        user_id = row['userID']

        #update tables of userIDs
        if user_id not in data:
            hostnames = {}
            for host in raw_hostnames:
                hostnames[host] = 1
    
            data[user_id] = hostnames
        else:
            for host in raw_hostnames:
                if host in data[user_id]:
                    data[user_id][host] += 1
                else:
                    data[user_id][host] = 1
    return data

# similarity from paper
def kopp_similarity(host_a, host_b):
    timewindow_num = 24.0
    common = list(set(host_a.keys() + host_b.keys()))
    
    #calculating similarity
    if len(common) == 0:
        res = 0
    else:
        a = 0.0
        b = 0.0
        c = 0.0
    
        for host in common:
            f_a = float(host_a.get(host, 0)) / timewindow_num
            f_b = float(host_b.get(host, 0)) / timewindow_num
                
            a += float(f_a) * float(f_b)
            b += float(f_a)**2
            c += float(f_b)**2
   
        if a == 0.0:
            res = 0.0
        else:
            res = float(a) / (np.sqrt(float(b)) * np.sqrt(float(c)))
    
    return res