#!/usr/bin/env python
# coding: utf-8

# In[1]:


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
import time

df = pd.read_csv("data.csv" , index_col=False)
print(df.head(5))


del df['Unnamed: 32']
df['diagnosis'] = df['diagnosis'].apply(lambda x : '1' if x == 'M' else '0')

fig = plt.figure()
ax1 = fig.add_subplot(111)
cmap = cm.get_cmap('jet', 10)
cax = ax1.imshow(df.corr(), interpolation="none", cmap=cmap)
plt.title('Correlation Between Breast Cancer Attributes')
plt.show()


# In[61]:


Y = df['diagnosis'].values
X = df.drop(['diagnosis'] , axis = 1).values

X_train , X_test , Y_train , Y_test  = train_test_split(X , Y , test_size = 0.2, random_state = 15)


# In[62]:


models_list = []
models_list.append(('CART', DecisionTreeClassifier()))
models_list.append(('SVM' , SVC()))
models_list.append(('NB' , GaussianNB()))
models_list.append(('KNN' , KNeighborsClassifier()))


# In[74]:


num_folds = 20
results = []
names = []

for name , model in models_list:
    kfold = KFold(n_splits = num_folds)
    start = time.time()
    cv_results = cross_val_score(model , X_train , Y_train , cv = kfold , scoring = 'accuracy')
    end = time.time()
    results.append(cv_results)
    names.append(name)
    print("%s: %f (%f) (run time: %f)" %(name, cv_results.mean(), cv_results.std(), end-start))


# In[83]:


import warnings 

pipelines = []

pipelines.append(('ScaledCART', Pipeline([('Scaler', StandardScaler()),('CART',
                                                                        DecisionTreeClassifier())])))
pipelines.append(('ScaledSVM', Pipeline([('Scaler', StandardScaler()),('SVM', SVC( ))])))
pipelines.append(('ScaledNB', Pipeline([('Scaler', StandardScaler()),('NB',
                                                                      GaussianNB())])))
pipelines.append(('ScaledKNN', Pipeline([('Scaler', StandardScaler()),('KNN',
                                                                       KNeighborsClassifier())])))

results = []
names = []
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    kfold = KFold(n_splits = num_folds)
    for name , model in pipelines:
        start = time.time()
        cv_results = cross_val_score(model , X_train , Y_train , cv = kfold , scoring = 'accuracy')
        end = time.time()
        results.append(cv_results)
        names.append(name)
        print("%s: %f (%f) (run time: %f)" % (name , cv_results.mean() , cv_results.std() , end-start))


# In[77]:


fig = plt.figure()
fig.suptitle('Performance Comparison')
ax = fig.add_subplot(111)
plt.boxplot(results)
ax.set_xticklabels(names)
plt.show()


# In[78]:


scaler = StandardScaler().fit(X_train)
rescaledX = scaler.transform(X_train)
c_values = [0.1 , 0.3 , 0.5 , 0.7 , 0.9 , 1.0 , 1.3 , 1.5 , 1.7 , 2.0]
kernel_values = ['linear' , 'poly' , 'rbf' , 'sigmoid']
param_grid = dict(C = c_values , kernel = kernel_values)
model = SVC()
kfold = KFold(n_splits = num_folds)
grid = GridSearchCV(estimator = model , param_grid = param_grid , scoring = 'accuracy' , cv = kfold)
grid_result = grid.fit(rescaledX , Y_train)
print("Best : %f using %s" % (grid_result.best_score_ , grid_result.best_params_))
means = grid_result.cv_results_['mean_test_score']
stds = grid_result.cv_results_['std_test_score']
params = grid_result.cv_results_['params']
for mean, stdev, param in zip(means, stds, params):
    print("%f (%f) with: %r" % (mean, stdev, param))


# In[79]:


with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    scaler = StandardScaler().fit(X_train)
X_train_scaled = scaler.transform(X_train)
model = SVC(C=2.0, kernel='rbf')
start = time.time()
model.fit(X_train_scaled, Y_train)
end = time.time()
print( "Run Time: %f" % (end-start))


# In[80]:


# estimate accuracy on test dataset
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    X_test_scaled = scaler.transform(X_test)
predictions = model.predict(X_test_scaled)


# In[81]:


print("Accuracy score %f" % accuracy_score(Y_test, predictions))
print(classification_report(Y_test, predictions))


# In[82]:


print(confusion_matrix(Y_test, predictions))


# In[ ]:




