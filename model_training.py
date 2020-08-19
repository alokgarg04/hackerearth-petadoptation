# -*- coding: utf-8 -*-
"""
Created on Thu Aug  6 16:01:42 2020

@author: 1094356
"""
import pandas as pd
import numpy as np
from DatapreProcessing import baisc_details
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier



def oneHotEncoder(X,column):
    the_one_hot_encodings = pd.get_dummies(X[column].values, prefix = 'Color',drop_first=True)
    #one_hot = OneHotEncoder(drop= 'if_binary').fit_transform(X[column].values.reshape(1, -1))
    #drop_binary_enc = OneHotEncoder(drop='if_binary').fit(X)
    #X[column] = one_hot
    return the_one_hot_encodings

def StandardScaler_(X,column_list):
    X[column_list] = StandardScaler().fit_transform(X[column_list])
    return X


def model_trainig(X,target):
    #X_train, X_test, y_train, y_test = train_test_split(X,target,test_size=0.2,random_state=42,stratify = target)
    #clf = RandomForestClassifier(random_state=42, n_jobs= -1,verbose=4)
    #clf = RandomForestClassifier(n_estimators=312,max_features= 'log2',max_depth=10,criterion='entropy')
    clf = RandomForestClassifier(n_estimators=189,max_features= 'log2',max_depth=14,criterion='gini')
    clf.fit(X,target)
    prediction = clf.predict(X)
    score = accuracy_score(target, prediction)
    return clf,score



def model_trainig_lr(X,target):
    lr = LogisticRegression(verbose=4)
    X_train, X_test, y_train, y_test = train_test_split(X,target,test_size=0.3,stratify = target)
    lr.fit(X_train,y_train)
    prediction = lr.predict(X_test)
    score = accuracy_score(y_test, prediction)
    return lr,score

def support_vector(X,target):
    X_train, X_test, y_train, y_test = train_test_split(X,target,test_size=0.3,stratify = target)
    rf = SVC(verbose=4)
    rf.fit(X_train,y_train)
    prediction = rf.predict(X_test)

    score = accuracy_score(y_test, prediction)
    return rf,score

def decisionTree(X,target):
    dt = DecisionTreeClassifier(criterion= 'gini',max_depth=19)
    dt.fit(X,target)
    return dt
def GradientBoosting(X,target):
    xgb = GradientBoostingClassifier(n_estimators= 250,verbose=4)
    xgb.fit(X,target)
    return xgb
if __name__== '__main__':
    print('training file')
    prepocessed_train = pd.read_csv('./Dataset/prepocessed_train.csv')
    print('---check basic details---')
    #baisc_details(prepocessed_train)
    target_variables = prepocessed_train.loc[:,['breed_category','pet_category']]
    prepocessed_train.drop(['breed_category','pet_category'],axis = 1,inplace = True)
    #X = prepocessed_train.iloc[:,:-2]

    X1 = oneHotEncoder(prepocessed_train,'color_type')
    X2 = pd.concat([prepocessed_train, X1 ] , axis = 1)
    #print(X1.head())
    X2.drop('color_type',axis = 1 ,inplace = True)
    #X2.to_csv('./Dataset/prepocessed_train_updated.csv',index = False)
    print('StandardScaler_ to the columns')
    column_list = ['length(m)', 'height(cm)','total_days']
    X2 = StandardScaler_(X2,column_list)
    x_final = X2.drop('X2',axis = 1)
    x_final.to_csv('./Dataset/prepocessed_train_updated.csv',index = False)
    # print(f"shape of X:{X.shape}")
    # print(f"shape of target_variables:{target_variables.shape}")
    breed_category = target_variables.iloc[:,-2]
    pet_category = target_variables.iloc[:,-1]
    # print(f"shape of breed_category:{breed_category.shape}")
    # print(f"shape of pet_category:{pet_category.shape}")


# =============================================================================
    print("training started")
    # breed_category_model,score_breed_category = model_trainig(X2,breed_category)
    # pet_category_model,score_pet_category = model_trainig(X2,pet_category)
    # breed_category_model = GradientBoosting(X2,breed_category)
    # pet_category_model = GradientBoosting(X2,pet_category)
    # joblib.dump(breed_category_model, './models/breed_category_17th.pkl')
    # joblib.dump(pet_category_model, './models/pet_category_17th.pkl')
    # print(f"score_breed_category:{score_breed_category}")
    # print(f"score_pet_category:{score_pet_category}")
# =============================================================================





# =============================================================================
    # print('traing in with logistic regression')
    # breed_category_model_lr,score_breed_category_lr = model_trainig_lr(X,breed_category)
#     pet_category_model_lr,score_pet_category_lr = model_trainig(X,pet_category)
#
    #print(f"score score_breed_category_lr:{score_breed_category_lr}")
#     print(f"score score_pet_category_lr:{score_pet_category_lr}")
# =============================================================================
    # modefl,score = model_trainig_lr(X2,pet_category)
    # print(f" score :{score}")