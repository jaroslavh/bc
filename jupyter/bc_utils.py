import pandas as pd
import numpy as np
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
            if similarity_measure(point, rep) <= delta: #here add distance measure as parameter
                break
        else:
            representatives = np.vstack((representatives, point))
            
    return pd.DataFrame(representatives, columns=['x', 'y'])
