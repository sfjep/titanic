# -*- coding: utf-8 -*-
"""
Created on Thu Mar  5 14:17:28 2020

@author: Simon
"""

#%% Libraries
import pandas as pd
import os
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score, KFold, GridSearchCV
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.metrics import accuracy_score, confusion_matrix
import pydotplus as pydot
#from Ipython.display import Image   # NOT WORKING (NEED FOR DRAWING DECISION TREE)
from io import StringIO
import statsmodels.api as sm
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
import xgboost as xgb


#%% IMPORT CSV
test_df = pd.read_csv('data/test.csv')
train_df = pd.read_csv('data/train.csv')
pid = test_df.PassengerId
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

#%% ADD FAMILY SIZE

train_df["family_size"] = train_df["SibSp"]+train_df["Parch"]+1
test_df["family_size"] = test_df["SibSp"]+test_df["Parch"]+1

# DROP INPUTS TO FAMILY SIZE
train_df.drop(['SibSp', 'Parch'], 1, inplace=True)
test_df.drop(['SibSp', 'Parch'], 1, inplace=True)

# ADD IS ALONE FEATURE
train_df['IsAlone'] = 1 #initialize to 1/yes
train_df['IsAlone'].loc[train_df['family_size'] > 1] = 0
test_df['IsAlone'] = 1 #initialize to 1/yes
test_df['IsAlone'].loc[test_df['family_size'] > 1] = 0

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

#%% FAMILY SIZE HYPOTHESIS

family_sizes = np.sort(train_df['family_size'].unique())

for i in family_sizes:
    families = train_df.loc[train_df['family_size']==i]
    numberOfFamilies = len(families)
    survival_rate = len(train_df.loc[(train_df['family_size']==i)&(train_df['Survived']==1)])/ numberOfFamilies
    print('Family Size {}: {} instances, {} survival rate'.format(i,numberOfFamilies,round(survival_rate, 2)))

sns.countplot(x='family_size', hue='Survived', data=train_df).set_title('Survival by family size')

#%% DEFINE FOUR FAMILY BINS

familyBinMapping = {1: 1, 2: 2, 3: 2, 4: 2, 5: 3, 6: 3, 7: 3, 8: 3, 11: 3}
train_df['familySizeBins'] = train_df['family_size'].map(familyBinMapping)
test_df['familySizeBins'] = test_df['family_size'].map(familyBinMapping)

train_df.drop('family_size',1,inplace=True)
test_df.drop('family_size',1,inplace=True)

sns.countplot(x='familySizeBins', hue='Survived', data=train_df).set_title('Survival by family size')


#%% DROP ALL UNNECESSARY DATA


train_df = train_df[['Survived', 'Pclass', 'Sex', 'Age', 'Fare', 'IsAlone', 'familySizeBins', 'PortEmb']]
X_test = test_df[['Pclass', 'Sex', 'Age', 'Fare', 'IsAlone', 'familySizeBins', 'PortEmb']]

#%% SPLIT INTO FEATURES AND LABELS

# ARRAYS
features = train_df.iloc[:,[1,2,3,4,5,6,7]].values
labels = train_df.iloc[:,0].values

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
features = sc.fit_transform(features)
X_test = sc.fit_transform(X_test)

#%% TESTING AND TRAIN SET

features_trn, features_tst, label_trn, label_tst = train_test_split(features, labels, test_size = 0.25, random_state = 42)


#%% CV ALGORITHM CHECK MODELS

pipelines = []
pipelines.append(('DecisionTree', Pipeline([('DecisionTree', tree.DecisionTreeClassifier())])))
pipelines.append(('LogisticRegression', Pipeline([('LogisticRegression', LogisticRegression())])))
pipelines.append(('RandomForest', Pipeline([('RF', RandomForestClassifier())])))
pipelines.append(('SupportVector', Pipeline([('SVC', SVC())])))
pipelines.append(('GradientBoosting', Pipeline([('GBM', GradientBoostingClassifier())])))
pipelines.append(('XGBoost', Pipeline([('XGB', xgb.XGBClassifier())])))

kfold = KFold(n_splits=5)

results = []
names = []

for name, model in pipelines:
    
    cv_results = cross_val_score(model, features_trn, label_trn, cv=kfold, scoring='neg_mean_absolute_error')
    results.append(cv_results)
    names.append(name)
    #print(cv_results)
    msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
    print(msg)


#%% MODEL TUNING

def randomForestTuning(train_features, train_labels, test_features):
        
    parameters = {'n_estimators': range(60, 160, 20), 
                  'max_depth': range(4, 16, 1),
                 }
    
    rf = RandomForestClassifier(criterion = 'entropy')
    
    np.random.seed(1)
    gs_rf = GridSearchCV(rf, parameters, scoring = 'accuracy', cv = kfold)
    gs_rf.fit(train_features, train_labels)
    
    rf_label_prd = gs_rf.best_estimator_.predict(test_features)
    print(gs_rf.best_estimator_)
    print(gs_rf.best_score_)   
    
    return rf_label_prd


def svcTuning(train_features, train_labels, test_features):
    parameters = {'kernel': ['linear', 'poly', 'rbf', 'sigmoid'], 
              'C': [0.01, 0.1, 0.5, 1, 1.5, 2]
              }

    svm = SVC(random_state = 0)
    
    np.random.seed(1)
    gs_svm = GridSearchCV(svm, parameters, scoring = 'accuracy', cv = kfold)
    gs_svm.fit(train_features, train_labels)
    
    svm_label_prd = gs_svm.best_estimator_.predict(test_features)
    print(gs_svm.best_estimator_)
    print(gs_svm.best_score_)

    return svm_label_prd
    
def gbTuning(train_features, train_labels, test_features):
    parameters = {'loss': ('deviance', 'exponential'), 
              'learning_rate': [0.001, 0.01, 0.05, 0.15],
              'n_estimators': range(60,160,20)
             }

    gb = GradientBoostingClassifier()
    
    np.random.seed(1)
    gs_gb = GridSearchCV(gb, parameters, scoring = 'accuracy', cv = kfold)
    gs_gb.fit(train_features, train_labels)
    
    gb_label_prd = gs_gb.best_estimator_.predict(test_features)
    print(gs_gb.best_estimator_)
    print(gs_gb.best_score_)
    
    return gb_label_prd
       
    
def xgbTuning(train_features, train_labels, test_features):
    parameters = {'booster': ['gbtree', 'gblinear', 'dart'],
                  'max_depth': range(4, 10, 1),
                  'learning_rate': [0.01, 0.05, 0.1, 0.5]}
     
    xgbc = xgb.XGBClassifier()
    np.random.seed(1)
    gs_xgb = GridSearchCV(xgbc, parameters, scoring = 'accuracy', cv = kfold)
    gs_xgb.fit(train_features, train_labels)
    
    xgb_label_prd = gs_xgb.best_estimator_.predict(test_features)
    print(gs_xgb.best_estimator_)
    print(gs_xgb.best_score_)
  
    return xgb_label_prd

def logregTuning(train_features, train_labels, test_features):
    parameters = {'C': np.arange(1, 2, 0.1),
                  'solver': ["newton-cg", "lbfgs", "liblinear", "sag", "saga"],
                  'fit_intercept': [True, False]
                  }      
    lr = LogisticRegression()
    np.random.seed(1)
    gs_lr = GridSearchCV(lr, parameters, scoring = 'accuracy', cv = kfold)
    gs_lr.fit(train_features, train_labels)
    
    lr_label_prd = gs_lr.best_estimator_.predict(test_features)
    print(gs_lr.best_estimator_)
    print(gs_lr.best_score_)
  
    return lr_label_prd

#%% MODEL TUNING

rf_train_pred = randomForestTuning(features_trn, label_trn, features_tst)
svc_train_pred = svcTuning(features_trn, label_trn, features_tst)
gb_train_pred = gbTuning(features_trn, label_trn, features_tst)
xgb_train_pred = xgbTuning(features_trn, label_trn, features_tst)
lr_train_pred = logregTuning(features_trn, label_trn, features_tst)


accuracy_score(label_tst, rf_train_pred)
accuracy_score(label_tst, svc_train_pred)
accuracy_score(label_tst, gb_train_pred)
accuracy_score(label_tst, xgb_train_pred)
accuracy_score(label_tst, lr_train_pred)

#%% TEST ENSEMBLES USING FULL DATA SET

rf_test_pred = randomForestTuning(features, labels, X_test)
svc_test_pred = svcTuning(features, labels, X_test)
gb_test_pred = gbTuning(features, labels, X_test)
xgb_test_pred = xgbTuning(features, labels, X_test)
lr_test_pred = logregTuning(features, labels, X_test)

#%%
comb5 = rf_test_pred + svc_test_pred + gb_test_pred + xgb_test_pred + lr_test_pred

out5 = np.where(comb5 < 3, 0, 1)

comb3 = rf_test_pred + svc_test_pred + xgb_test_pred

out3 = np.where(comb3 < 2, 0, 1)


#%% SUBMISSION COMB5

submission = pd.DataFrame({
        "PassengerId": pid,
        "Survived": out5
    })

submission.to_csv('out_comb5.csv', index = False)

# Score: 0.78468

#%% SUBMISSION COMB3

submission = pd.DataFrame({
        "PassengerId": pid,
        "Survived": out3
    })

submission.to_csv('out_comb3.csv', index = False)

# Score: 0.78708

#%% SUBMISSION gb

submission = pd.DataFrame({
        "PassengerId": pid,
        "Survived": gb_test_pred
    })



submission.to_csv('out_gb.csv', index = False)

# Score: 0.76794

#%% SUBMISSION xgb

submission = pd.DataFrame({
        "PassengerId": pid,
        "Survived": xgb_test_pred
    })



submission.to_csv('out_xgb.csv', index = False)

# Score: 0.77272

#%% SUBMISSION svc

submission = pd.DataFrame({
        "PassengerId": pid,
        "Survived": svc_test_pred
    })



submission.to_csv('out_svc.csv', index = False)

# Score: 0.78229

#%% SUBMISSION rf

submission = pd.DataFrame({
        "PassengerId": pid,
        "Survived": rf_test_pred
    })

submission.to_csv('out_rf.csv', index = False)

# Score: 0.79665


#%% SUBMISSION log reg

submission = pd.DataFrame({
        "PassengerId": pid,
        "Survived": lr_test_pred
    })

submission.to_csv('out_lr.csv', index = False)

# Score: 0.76076

#%% RANDOM FOREST FEATURE IMPORTANCE

f_imp = gs_rf.best_estimator_.feature_importances_

cn = train_df.iloc[:,1:]
f_names = cn.columns

f_df = pd.concat([pd.Series(f_names), pd.Series(f_imp)], axis = 1)
f_df.columns = ['Feature', 'Variable Importance']
print(f_df.sort_values('Variable Importance', ascending = False))
