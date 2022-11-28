#------------------------------------------------ABOUT DATASET

#The dataset contains transactions made by credit cards in September 2013 by European cardholders.
#This dataset presents transactions that occurred in two days, where we have 492 frauds out of 284,807 transactions. 
# The dataset is highly unbalanced, the positive class (frauds) account for 0.172% of all transactions.
#It contains only numerical input variables which are the result of a PCA transformation.
#  Unfortunately, due to confidentiality issues, we cannot provide the original features and more background information about the data. 
# Features V1, V2, … V28 are the principal components obtained with PCA, the only features which have not been transformed with PCA are 'Time' and 'Amount'. 
# Feature 'Time' contains the seconds elapsed between each transaction and the first transaction in the dataset. 
# The feature 'Amount' is the transaction Amount, this feature can be used for example-dependant cost-sensitive learning.
# Feature 'Class' is the response variable and it takes value 1 in case of fraud and 0 otherwise.
#Given the class imbalance ratio, we recommend measuring the accuracy using the Area Under the Precision-Recall Curve (AUPRC). 
# Confusion matrix accuracy is not meaningful for unbalanced classification.

#------------------------------------------------work flow

#data collection
#data anlysis
#data separation
#data train test split
#model use
#prediction

#------------------------------------------------import labrary

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression

#---------------------------------------------------data anlysis

data = pd.read_csv("C:/Users/kunde/all vs code/ml prject/creditcard.csv")
print(data.shape)
print(data.columns)
print(data.info())
print(data.describe())
print(data.isnull().sum())
print(data.head(5))
print(data.tail(5))
print(data['Class'].value_counts())

#----------------------------------------------------data separation

x = data.drop(columns=["Time", "Class"], axis=1)
y = data["Class"]

#---------------------------------------------------train-test-split

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, stratify=y, random_state=2)
print(x.shape, x_train.shape, x_test.shape)
print(y.shape, y_train.shape, y_test.shape)

#---------------------------------------------------model use
model = LogisticRegression()
model.fit(x_train, y_train)

#--------------------------------------------predict train data 
y_train_p = model.predict(x_train)
print(y_train_p, "THIS IS OUR DATA", np.array(y_train), "THIS IS TRUE DATA")
accur = accuracy_score(y_train_p, y_train)
print(accur)

#-------------------------------------------predict test data

y_test_p = model.predict(x_test)
print(y_test_p, "this is our data", np.array(y_test), "this is true data")
acur = accuracy_score(y_test, y_test_p)
print(accur, "this is test data accurancy")

#-------------------------------------------this is new single data prediction

x = [-1.3598071336738,-0.0727811733098497,2.53634673796914,1.37815522427443,-0.338320769942518,0.462387777762292,0.239598554061257,0.0986979012610507,0.363786969611213,0.0907941719789316,-0.551599533260813,-0.617800855762348,-0.991389847235408,-0.311169353699879,1.46817697209427,-0.470400525259478,0.207971241929242,0.0257905801985591,0.403992960255733,0.251412098239705,-0.018306777944153,0.277837575558899,-0.110473910188767,0.0669280749146731,0.128539358273528,-0.189114843888824,0.133558376740387,-0.0210530534538215,149.62]

x_new = np.asarray(x)
x_predict = x_new.reshape(1,-1)
y_predict = model.predict(x_predict)
print(y_predict)










