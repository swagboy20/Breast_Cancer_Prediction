import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from matplotlib import cm as cm
import warnings
import time

from Breast_Cancer_Prediction.py import *

def predict(df):
  Y = df['diagnosis'].values
  X = df.drop(['diagnosis'] , axis = 1).values
  with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    X_scaled = scaler.transform(X)
  predictions = model.predict(X_scaled)
  return predictions
 
  
  
