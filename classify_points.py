# Finding out to what cluster the point belongs.

import sys
import argparse
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier 
from sklearn.metrics import classification_report, confusion_matrix  

#--------------FUNCTIONS START ------------------------------------------------

#returns tuple (ids, data) for further usage 
def readDataAndIDs(filePath):
    reader = pd.read_csv(filePath, names=names)

    ids = reader.iloc[:, 0].values
    data = reader.iloc[:, 1:].values
    return [data, ids]


#--------------FUNCTIONS END ------------------------------------------------


parser = argparse.ArgumentParser()
parser.add_argument("inFile", help="csv file to read unclassified points from")
parser.add_argument("medoidFile", help="csv file to read medoid data from")
#TODO add euclidean vs cosine difference
args = parser.parse_args()
pInFile = args.inFile
pMedoidFile = args.medoidFile

# Assign colum names to the dataset
names = ['cluster', 'x', 'y', 'z']

# Read dataset to pandas dataframe
medoids = readDataAndIDs(pMedoidFile) #x_train, y_train
unclassified = readDataAndIDs(pInFile) #x_test, y_test

x_train = medoids[0]
y_train = medoids[1]
x_test = unclassified[0]
y_test = unclassified[1]

scaler = StandardScaler() #TODO read how this works
scaler.fit(x_train)

x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

classifier = KNeighborsClassifier(n_neighbors=5)  
classifier.fit(x_train, y_train)

y_pred = classifier.predict(x_test)

print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))