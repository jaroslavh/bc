import pandas as pd
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
