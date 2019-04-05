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

    #here starts RepAssign routine for advanced delta-medoids
    for row in df.iterrows():
        point = tuple(row[1])

        for rep in representatives: #needs optimalization
            if similarity_measure(point, rep) <= delta:
                break
        else:
            representatives = np.vstack((representatives, point))

    return pd.DataFrame(representatives, columns=df.columns.values)

def delta_medoids(df, delta, similarity_measure):
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
    clusters = {}

    #here starts RepAssign routine for advanced delta-medoids
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
            clusters[representative] = np.vstack((clusters[representative],
                                                point))
        else:
            clusters[point] = np.array(point, ndmin=2)

    return pd.DataFrame(clusters.keys(), columns=df.columns.values)

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
    t = 0
    representatives = {}
    representatives[t] = set()

    while True:
        #print("\n=========== running for t = " + str(t) + "============")
        clusters = {}
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
            for index in index_list:
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
    #print(classification_report(y_test, y_pred))
    return conf_mat

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
