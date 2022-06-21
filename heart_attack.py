# -*- coding: utf-8 -*-
"""
Created on Tue Jun 21 09:20:38 2022

@author: Calvin
"""
import os
import pickle
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder

# static
CSV_PATH = os.path.join(os.getcwd(),'data','heart.csv')
BEST_PIPE_PATH = os.path.join(os.getcwd(),'model','best_pipe.pkl')
BEST_MODEL_PATH = os.path.join(os.getcwd(),'model','best_model.pkl')

# EDA
# Step 1) Data Loading
df = pd.read_csv(CSV_PATH)

# Step 2) Data Inspection
df.info()
df.describe().T
df.isna().sum() # no NaNs
df.duplicated().sum() # identified 1 duplicate data

plt.figure(figsize=(20,20))
df.boxplot()
plt.show() 
# Based on the boxplot, there are outlier in 'trtbps','chol','thalachh','fbs'
#'oldpeak','caa'

# Step 3) Data Cleaning
df = df.drop_duplicates()
df.duplicated().sum() # verify to ensure the duplicate data is removed.
df.isna().sum() # verify to ensure there is no NaNs after removing duplicate.
df.info()
# No further cleaning needed

# continous columns
con_column = df.loc[:,['age','trtbps','chol','thalachh','oldpeak']]

# category columns
cat_column = df.loc[:,['sex','cp','fbs','restecg','exng','slp','caa','thall',
                       'output']]

# plot con_column
for con in con_column:
    plt.figure()
    sns.distplot(df[con])
    plt.show()

# plot cat_column
for cat in cat_column:
    plt.figure()
    sns.countplot(df[cat])
    plt.show()

le = LabelEncoder()
df['output'] = le.fit_transform(df['output'])
LABEL_ENCODER_PATH = os.path.join(os.getcwd(),'model','Label_Encoder.pkl')
with open(LABEL_ENCODER_PATH,'wb') as file:
    pickle.dump(le,file)

# Step 4) Features Selection
# finding correlation/relationship between cont vs cat or feature vs target.
X = df.drop(labels=['output'],axis=1)
y = df['output']

for i in X.columns:
    print(i)
    lr = LogisticRegression(solver='liblinear')
    lr.fit(np.expand_dims(X[i],axis=-1),y)
    print(lr.score(np.expand_dims(X[i],axis=-1),y))
    
    if lr.score(np.expand_dims(X[i],axis=-1),y) <0.5:
        X = X.drop(labels=[i],axis=1)

# based on the logistic regression accuracy, all feautures are selected because
# they have accuracy more than 50%
# new_X = X = df.drop(labels=['output'],axis=1)

# Step 5) Data Preprocessing
# splitting dataset
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3,
                                                  random_state=123)

#%% Pipeline Creation
# 1) Determine whether MMS or SS is better in this case and  which classifier 
# works the best in this case
# Logistic Regression,Decision Tree,RandomForest,KNN,SVC

# Logistic regression pipeline
std_lr = Pipeline([('StandardScaler', StandardScaler()),
                      ('LogisticClassifier', LogisticRegression(solver='liblinear'))])

mms_lr = Pipeline([('MinMaxScaler', MinMaxScaler()),
                      ('LogisticClassifier', LogisticRegression(solver='liblinear'))])
                                                                    
# KNN pipeline
std_knn = Pipeline([('StandardScaler', StandardScaler()),
                       ('KNNClassifier', KNeighborsClassifier())])

mms_knn = Pipeline([('MinMaxScaler', MinMaxScaler()),
                       ('KNNClassifier', KNeighborsClassifier())])
                                                                  
# RF pipeline
std_rf = Pipeline([('StandardScaler', StandardScaler()),
                      ('RFClassifier', RandomForestClassifier())])

mms_rf = Pipeline([('MinMaxScaler', MinMaxScaler()),
                      ('RFClassifier', RandomForestClassifier())])
                                                                  
# DT pipeline
std_dt = Pipeline([('StandardScaler', StandardScaler()),
                      ('DTClassifier', DecisionTreeClassifier())])

mms_dt = Pipeline([('MinMaxScaler', MinMaxScaler()),
                      ('DTClassifier', DecisionTreeClassifier())])
                                                                  
# SVC pipeline
std_svc = Pipeline([('StandardScaler', StandardScaler()),
                       ('SVClassifier', SVC())])

mms_svc = Pipeline([('MinMaxScaler', MinMaxScaler()),
                       ('SVClassifier', SVC())])

# Create a list for the pipeline so that it can be iterate
pipelines = [std_lr, mms_lr, std_knn, mms_knn, std_rf, 
             mms_rf, std_dt, mms_dt, std_svc, mms_svc, 
             mms_svc]

for pipeline in pipelines:
    pipeline.fit(X_train,y_train)

#%% Pipeline Analysis/Model Evaluation
best_accuracy = 0
pipeline_scored = []

# using for&enumerate to get scare in pipelines list
for i, pipeline in enumerate(pipelines):
    print(pipeline.score(X_test,y_test))
    pipeline_scored.append(pipeline.score(X_test,y_test))

best_pipeline = pipelines[np.argmax(pipeline_scored)] # to get highest score
best_accuracy = pipeline_scored[np.argmax(pipeline_scored)] # to get accuracy
print('The best combination of the pipeline is {} with accuracy of {}'
      .format(best_pipeline.steps,best_accuracy))

#%% GridSearch CV
# from the pipeline above, it is deduced that the pipeline with SS+KNN
# Achieved the highest accuracy when tested against test dataset

std_knn = Pipeline([('StandardScaler', StandardScaler()),
                    ('KNNClassifier', KNeighborsClassifier())])

grid_param = [{'KNNClassifier__n_neighbors':[5,10,50,100],
             'KNNClassifier__algorithm':['auto','ball_tree','kd_tree','brute'],
             'KNNClassifier__leaf_size':[30,60,90,120],
              'StandardScaler__with_mean':[True,False],
              'StandardScaler__with_std':[True,False]}]

# Using GridSearch or RandomizedSearchCV the accuracy result is still the same
grid_search = GridSearchCV(std_knn,grid_param,cv=5,verbose=1,n_jobs=-1)
best_model = grid_search.fit(X_train,y_train)

print(best_model.score(X_test,y_test))
print(best_model.best_index_)
print(best_model.best_params_)

with open(BEST_PIPE_PATH,'wb') as file:
    pickle.dump(best_model,file)

#%% Retrain the model with selected params so that result is the same and fine 
# tune is saved.
std_knn = Pipeline([('StandardScaler', StandardScaler(with_mean=(True),
                                                      with_std=(True))),
                    ('KNNClassifier', KNeighborsClassifier(algorithm='auto',
                                                           leaf_size=30,
                                                           n_neighbors=5))])

std_knn.fit(X_train,y_train)

with open(BEST_MODEL_PATH,'wb') as file:
    pickle.dump(std_knn,file)

#%% Model Analysis
y_true = y_test
y_pred = best_model.predict(X_test)

print(classification_report(y_true,y_pred))
print(confusion_matrix(y_true,y_pred))
print(accuracy_score(y_true,y_pred))

cf_matrix = confusion_matrix(y_true,y_pred)
group_names = ['True Neg','False Pos','False Neg','True Pos']
group_counts = ["{0:0.0f}".format(value) for value in cf_matrix.flatten()]
group_percentages = ["{0:.2%}".format(value) for value in 
                     cf_matrix.flatten()/np.sum(cf_matrix)]

labels = [f"{v1}\n{v2}\n{v3}" for v1, v2, v3 in 
          zip(group_names,group_counts,group_percentages)]
labels = np.asarray(labels).reshape(2,2)
sns.heatmap(cf_matrix, annot=labels, fmt='', cmap='Blues')

#%% Discussion
# Based on the description some 0 value represent null so it won't be remove to
# prevent data loss. The features selected is all but output which is y/target


