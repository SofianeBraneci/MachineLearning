# -*- coding: utf-8 -*-
"""
Created on Mon Apr 13 14:06:52 2020

@author: hp
"""

# importing libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# importing the data

dataset = pd.read_csv('Data.csv')

X = dataset.iloc[:, :-1].values
Y = dataset.iloc[:,-1].values

# taking care of missing values
from sklearn.preprocessing import Imputer
imputer = Imputer(missing_values='NaN', strategy='mean')
imputer = imputer.fit(X[:, 1:])
X[:, 1:] = imputer.transform(X[:, 1:])

# Encoding categorical data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
label_encoder_X = LabelEncoder()
X[:, 0] = label_encoder_X.fit_transform(X[:, 0])
# Dummy Encoding
encoder = OneHotEncoder(categorical_features=[0])
X = encoder.fit_transform(X).toarray()
Y = LabelEncoder().fit_transform(Y)

# Spliting the dataset 
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=0)

from sklearn.preprocessing import StandardScaler
scaler =  StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)