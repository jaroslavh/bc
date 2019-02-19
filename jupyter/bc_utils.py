import pandas as pd
import numpy as np
import random

from scipy.spatial import distance
from matplotlib import pyplot


# utils for usage in multiple other notebooks

class TestDf:
    train_df = None
    test_df = None

    def __init__(self, init_df):

        split_point = int(len(init_df) * 0.8)

        #splitting the dataset into train (80%) and test (20%) part
        self.train_df = init_df.iloc[:split_point]
        self.test_df = init_df.iloc[split_point:]

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

    print("delta_medoids_full algorithm ended after " + str(t) + " iterations.")
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

#using random select algorithm
def random_selection(df, delta, similarity_measure, num):
    rows = random.sample(df.index, num)
    representatives = df.loc[rows]
    return representatives
