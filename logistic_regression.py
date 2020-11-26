# data analysis and wrangling
import pandas as pd
import numpy as np
import random as rnd
import csv
# visualization libraries 
#import seaborn as sns
import matplotlib.pyplot as pltmatplotlibinline

# sklearn modules
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import RFE
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import Perceptron
from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2

'''
Dataset taken from kaggle
train.csv - https://www.kaggle.com/c/titanic/data/
test.csv  - https://www.kaggle.com/c/titanic/data/

'''

train_df = pd.read_csv("U:\Python_codes\Datasets\\train.csv")
test_df = pd.read_csv("U:\Python_codes\Datasets\\test.csv")
l1 = train_df.columns.values
l2 = test_df.columns.values
print ("The number of null or unvalid values are")
for value in l1:
    Number_Nan = train_df[value].isna()
    print (value,"- number of Nan are ",Number_Nan.sum()) #Number of null values

train_df.info() #gives info about the training dataset such as max,min values for each dataset.
for value in l2:
    Number_Nan = test_df[value].isna()
    print (value,"- number of Nan are ",Number_Nan.sum())
test_df.info()
print (len(test_df))

pid = test_df['PassengerId']

test_df = test_df[['Pclass','Sex','Embarked']]
test_df['Sex'].replace( 'female',0,inplace=True)
test_df['Sex'].replace('male',1,inplace=True)

test_df['Embarked'].replace( 'S',1,inplace=True)
test_df['Embarked'].replace('C',2,inplace=True)
test_df['Embarked'].replace('Q',3,inplace=True)


train_df = train_df.drop(columns = ['Ticket','Fare','Cabin'])
mean = train_df['Age'].mean()
train_df['Age'] = train_df['Age'].fillna(mean)
train_df['Embarked'] = train_df['Embarked'].fillna('Q')
l3 = train_df.columns.values
for value in l2:
    Number_Nan = train_df[value].isna()
    print (Number_Nan.sum())   
label = train_df['Survived']
features = train_df.drop(columns = ['PassengerId','Survived','Name'])

features['Sex'].replace( 'female',0,inplace=True)
features['Sex'].replace('male',1,inplace=True)

features['Embarked'].replace( 'S',1,inplace=True)
features['Embarked'].replace('C',2,inplace=True)
features['Embarked'].replace('Q',3,inplace=True)

#To find how the features are ranked

model = LogisticRegression()
rfe = RFE(model, 3)
fit = rfe.fit(features,label)
print("Num Features: %s" % (fit.n_features_))
print("Selected Features: %s" % (fit.support_))
print("Feature Ranking: %s" % (fit.ranking_))

#Final features and predicting the solutions
fin_features = features.drop(columns = ['Age','SibSp','Parch'])
clf = GaussianNB()
clf.fit(fin_features,label)
predicted_labels = clf.predict(test_df)
#print ("FINISHED classifying. accuracy score : ")
#print (accuracy_score(test_labels, predicted_labels))
print (predicted_labels)

