# -*- coding: utf-8 -*-
"""
Created on Sun Jul 18 18:29:03 2021

@author: ankon
"""
#%%Importing modules and data
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import plot_tree

df = pd.read_csv('TASK-6 DATA.csv')

print("Info on training examples:")
print(df.info())

print("First 5 samples:\n",df.head())
print("Last 5 samples:\n",df.tail())

#%%Data Cleaning

#Checking for null values
print("Null Values:")
print(df.isnull().sum())

#Checking for duplicate values
dup_rows_df=df[df.duplicated()]
print("\nNo. of duplicate rows:",dup_rows_df.shape[0])

#%%Fitting the tree
X_np=df[['SepalLengthCm','SepalWidthCm','PetalLengthCm','PetalWidthCm']].to_numpy()
#print(X_np)
Y_np=df['Species']
#print(Y_np)

IrisDecTree=DecisionTreeClassifier()
IrisDecTree=IrisDecTree.fit(X_np,Y_np)

#%%Visualization
IrisFeatures=['SepalLengthCm','SepalWidthCm','PetalLengthCm','PetalWidthCm']
IrisLabel=['Iris-setosa', 'Iris-versicolor', 'Iris-virginica']
fig, axes = plt.subplots(nrows = 1,ncols = 1,figsize = (4,4), dpi=300)
plot_tree(IrisDecTree,
               feature_names=IrisFeatures, 
               class_names=IrisLabel,
               filled = True);
fig.savefig('IrisDecisionTree.png')
