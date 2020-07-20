# -*- coding: utf-8 -*-
"""
Created on Thu Mar  5 14:17:28 2020

@author: Simon
"""

#%% Libraries
import pandas as pd
import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn import tree
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
import pydotplus as pydot
#from Ipython.display import Image   # NOT WORKING (NEED FOR DRAWING DECISION TREE)
from io import StringIO
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm

os.chdir('C:\\Users\\Simon\\Desktop\\Data Science\\Kaggle\\Titanic')

#%% IMPORT CSV
test_df = pd.read_csv('data/test.csv')
train_df = pd.read_csv('data/train.csv')

#%% SUMMARY STATS

train_df.describe()

# 0 if survived, 1 if died, 38% survived
# 177 age values missing (We use feature engineering to impute missing age)

test_df.describe()

#%% FEATURE ENGINEERING TRAIN

honorifics = []

# We split each element at the comma, take the second element, then at period, 
# take the first, and remove all spacing using strip()

for name in train_df['Name']:
    honorifics.append(name.split(',')[1].split('.')[0].strip())

train_df['Honorific'] = honorifics

train_df.head(5)
train_df['Honorific'].value_counts()

#%% FEATURE ENGINEERING TRAIN

honorifics = []

# We split each element at the comma, take the second element, then at period, 
# take the first, and remove all spacing using strip()

for name in test_df['Name']:
    honorifics.append(name.split(',')[1].split('.')[0].strip())

test_df['Honorific'] = honorifics

test_df.head(5)
test_df['Honorific'].value_counts()

#%% Normalize Honorifics

honorific_map = {
        'Capt': 'Seviceman',
        'Col': 'Seviceman',
        'Don': 'Sir',
        'Dr': 'Seviceman',
        'Jonkheer': 'Sir',
        'Lady': 'Lady',
        'Major': 'Seviceman',
        'Master': 'Master',
        'Miss': 'Miss',
        'Mlle': 'Miss',
        'Mme': 'Mrs',
        'Mr': 'Mr',
        'Mrs': 'Mrs',
        'Ms': 'Mrs',
        'Rev': 'Seviceman',
        'Sir': 'Sir',
        'the Countess': 'Lady',
        'Dona': 'Miss'
        }

train_df['Washed Honorific'] = train_df.Honorific.map(honorific_map)
train_df['Washed Honorific'].value_counts()

train_df.boxplot(column='Age', by ='Washed Honorific', grid = True, figsize = (10,7))


test_df['Washed Honorific'] = test_df.Honorific.map(honorific_map)
test_df['Washed Honorific'].value_counts()

test_df.boxplot(column='Age', by ='Washed Honorific', grid = True, figsize = (10,7))

#%% FILL MISSING AGE VALUES WITH MEDIAN OF WASHED HONORIFIC TRAIN

train_df['Age'].fillna(-1,inplace = True)

honorifics = train_df['Washed Honorific'].unique()

medians = dict()

for honorific in honorifics:
    median = train_df.Age[(train_df['Age'] != -1) & (train_df['Washed Honorific'] == honorific)].median()
    medians[honorific] = median

for index , row in train_df.iterrows():
    if row['Age'] == -1:
        train_df.loc[index, 'Age'] = medians[row['Washed Honorific']]

#%%  FILL MISSING AGE VALUES WITH MEDIAN OF WASHED HONORIFIC TEST
test_df['Age'].fillna(-1,inplace = True)

honorifics = train_df['Washed Honorific'].unique()

medians = dict()

for honorific in honorifics:
    median = test_df.Age[(train_df['Age'] != -1) & (train_df['Washed Honorific'] == honorific)].median()
    medians[honorific] = median

for index , row in test_df.iterrows():
    if row['Age'] == -1:
        test_df.loc[index, 'Age'] = medians[row['Washed Honorific']]


#%%   FIX SEX CATEGORY
        
gendermap = {'male': 0, 'female': 1}
train_df['Sex'] = train_df['Sex'].map(gendermap)
test_df['Sex'] = test_df['Sex'].map(gendermap)

#%% EMBARKMENT ADJUSTMENT

emb = []

for place in train_df['Embarked']:
    if place == 'S':
        emb.append(1)
    elif place == 'C':
        emb.append(2)
    else:
        emb.append(3)

train_df['PortEmb'] = emb


emb = []

for place in test_df['Embarked']:
    if place == 'S':
        emb.append(1)
    elif place == 'C':
        emb.append(2)
    else:
        emb.append(3)

test_df['PortEmb'] = emb

#%% FARE ADJUSTMENT TRAIN

train_df['Fare'].fillna(-1,inplace = True)

pclass = train_df['Pclass'].unique()

medians = dict()

for pc in pclass:
    median = train_df.Fare[(train_df['Fare'] != -1) & (train_df['Pclass'] == pc)].median()
    medians[pc] = median

for index , row in train_df.iterrows():
    if row['Fare'] == -1:
        train_df.loc[index, 'Fare'] = medians[row['Pclass']]

#%% FARE ADJUSTMENT TRAIN
        
test_df['Fare'].fillna(-1,inplace = True)

pclass = test_df['Pclass'].unique()

medians = dict()

for pc in pclass:
    median = test_df.Fare[(test_df['Fare'] != -1) & (test_df['Pclass'] == pc)].median()
    medians[pc] = median

for index , row in test_df.iterrows():
    if row['Fare'] == -1:
        test_df.loc[index, 'Fare'] = medians[row['Pclass']]


#%% HYPOTHESIS

#Survivability against class
sns.countplot(x = 'Pclass', hue = 'Survived', data = train_df)

#Survivability against gender
sns.countplot(x = 'Sex', hue = 'Survived', data = train_df)

# Survivability against age
who = []

for age in train_df['Age']:
    if age > 8:
        who.append('Adult')
    else:
        who.append('Child')

train_df['Who'] = who

sns.countplot(x = 'Who', hue = 'Survived', data = train_df)




#%% DROP ALL UNNECESSARY DATA

titanic_df = train_df[['Survived', 'Pclass', 'Sex', 'Age', 'Fare', 'Parch', 'SibSp', 'PortEmb']]

test = test_df[['Pclass', 'Sex', 'Age', 'Fare', 'Parch', 'SibSp', 'PortEmb']]



#%% SPLIT INTO FEATURES AND LABELS

# ARRAYS
features = titanic_df.iloc[:,[1,2,3,4,5,6,7]].values
labels = titanic_df.iloc[:,0].values

#%% TESTING AND TRAIN SET

features_trn, features_tst, label_trn, label_tst = train_test_split(features, labels, test_size = 0.25, random_state = 42)

#%% DECISION TREE CLASSIFIER

dt = tree.DecisionTreeClassifier(max_depth = 7)
dt.fit(features_trn, label_trn)

dt_label_prd = dt.predict(features_tst)
accuracy_score(label_tst, dt_label_prd)

#%% DRAW DECISION TREE

dot_data = StringIO()
tree.export_graphviz(dt, out_file=dot_data, filled=True,
                     feature_names=list(titanic_df.drop(['Survived'], axis = 1).columns),
                     class_names=['Died', 'Survived'])

graph = pydot.graph_from_dot_data(dot_data.getvalue())
Image(graph.create_png())

#%% DECISION TREE MODEL OUTPUT

results = test_df[['PassengerId']]
results['Survived'] = pd.DataFrame(dt.predict(test))

results.to_csv('out_dt.csv', index = False)

#%% LOGISTIC REGRESSION

logreg = LogisticRegression()

logreg.fit(features_trn, label_trn)
logreg_label_prd = logreg.predict(features_tst)
accuracy_score(label_tst, logreg_label_prd)

#%% LOGISTIC REGRESSION MODEL OUTPUT

results = test_df[['PassengerId']]
results['Survived'] = pd.DataFrame(logreg.predict(test))

results.to_csv('out_logreg.csv', index = False)




