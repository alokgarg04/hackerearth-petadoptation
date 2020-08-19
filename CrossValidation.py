# -*- coding: utf-8 -*-
"""
Created on Thu Aug  6 16:46:48 2020

@author: 1094356
"""
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from scipy import stats
from sklearn.metrics import accuracy_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import roc_auc_score
from sklearn.metrics import classification_report
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier

def best_estimator_rf(X,target):
    #X_train, X_test, y_train, y_test = train_test_split(X,target,test_size=0.33,random_state=42,stratify = target)
    rfc = RandomForestClassifier(random_state=42)

    n_estimators = [int(x) for x in np.linspace(start = 100, stop = 500, num = 50)]
    max_features = ['auto', 'sqrt','log2']
    max_depth = [int(x) for x in np.linspace(10, 20, num = 6)]
    max_depth.append(None)



    random_grid = {'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'criterion' :['gini','entropy']
               }

    rf_random = RandomizedSearchCV(estimator = rfc, param_distributions = random_grid, n_iter = 100, cv = 5, verbose=4, random_state=42, n_jobs = -1)


    rf_random.fit(X,target)
    print(f"rf_random.best_params_:{rf_random.best_params_}")
    results_df = pd.DataFrame(rf_random.cv_results_)
    return results_df



'''
def cherchez(X,target,estimator, param_grid, search):

    """
    This is a helper function for tuning hyperparameters using teh two search methods.
    Methods must be GridSearchCV or RandomizedSearchCV.
    Inputs:
        estimator: Logistic regression, SVM, KNN, etc
        param_grid: Range of parameters to search
        search: Grid search or Randomized search
    Output:
        Returns the estimator instance, clf

    """
    X_train, X_test, y_train, y_test = train_test_split(X,target,test_size=0.33,random_state=42,stratify = target)
    try:
        if search == "grid":
            clf = GridSearchCV(
                estimator=estimator,
                param_grid=param_grid,
                scoring=None,
                n_jobs=-1,
                cv=10,
                verbose=4,
                return_train_score=True
            )
        elif search == "random":
            clf = RandomizedSearchCV(
                estimator=estimator,
                param_distributions=param_grid,
                n_iter=10,
                n_jobs=-1,
                cv=10,
                verbose=4,
                random_state=1,
                return_train_score=True
            )
    except:
        print('Search argument has to be "grid" or "random"')


    # Fit the model
    # clf.fit(X=X_train, y=y_train)
    # prediction = clf.predict(X_test)
    # score = accuracy_score(y_test,prediction)

    return clf'''

def hyperParameter_Lr(X,target):
    C = [1,5,10]
    penalty = ['l1', 'l2']
    #solver = ['liblinear', 'saga']
    hyperparameters = dict(C=C, penalty=penalty)
    logistic = LogisticRegression()
    gridsearch = GridSearchCV(logistic, hyperparameters,cv= 3,verbose=4)
    best_model = gridsearch.fit(X, target)
    print(best_model.best_estimator_)
    #return results_df_lr


def Knn_hyperParameter(X,target):
    X_train, X_test, y_train, y_test = train_test_split(X,target,test_size=0.33,random_state=42,stratify = target)
    knn = KNeighborsClassifier(n_jobs=-1)
    print('================================')
    #Training the model.
    knn.fit(X_train, y_train)
    #Predict test data set.
    y_pred = knn.predict(X_test)
    #Checking performance our model with classification report.
    print(classification_report(y_test, y_pred))
    #Checking performance our model with ROC Score.
    print(roc_auc_score(y_test, y_pred))
    print('===========================')

def DecisionTreeHyperParameter(X,target):
    dt = DecisionTreeClassifier()
    criterion = ['gini', 'entropy']
    max_depth = [i for i in range(4,20,3)]
    parameters = dict(criterion=criterion,max_depth=max_depth)
    # gridsearch = GridSearchCV(dt, parameters,cv=10,verbose=4,)
    randomized = RandomizedSearchCV(estimator = dt,param_distributions= parameters,cv=5,verbose=4)
    best_model = randomized.fit(X, target)
    #results_dt_df = pd.DataFrame(best_model.cv_results_)
    print(best_model.best_estimator_)
    print(best_model.best_params_)
    results_dt_df = pd.DataFrame(best_model.cv_results_)
    return results_dt_df

    # best_model = gridsearch.fit(X, target)
    # results_dt_df = pd.DataFrame(best_model.cv_results_)
    # print(best_model.best_estimator_)

def GradientBoosting(X,target):
    X_train, X_test, y_train, y_test = train_test_split(X,target,test_size=0.33,random_state=42,stratify = target)
    xgb = GradientBoostingClassifier(verbose=4,n_estimators=150)
    xgb.fit(X_train,y_train)
    #Predict test data set.
    y_pred = xgb.predict(X_test)
    #Checking performance our model with classification report.
    #print(classification_report(y_test, y_pred))
    #Checking performance our model with ROC Score.
    score = accuracy_score(y_test, y_pred)
    print(score)
    print('===========================')



if __name__== '__main__':
    print('started cross validation')
    prepocessed_train = pd.read_csv("E:/Life/HackerEarth_petAdoptation/Dataset\prepocessed_train.csv")
    prepocessed_train_updated = pd.read_csv('./Dataset/prepocessed_train_updated.csv')
    #X = prepocessed_train.iloc[:,:-2]

    target_variables = prepocessed_train.loc[:,['breed_category','pet_category']]
    x = prepocessed_train.drop(['breed_category','pet_category'],axis = 1)
    # print(f"shape of X:{prepocessed_train_updated.shape}")
    # print(f"shape of target_variables:{target_variables.shape}")

    breed_category = target_variables.iloc[:,-2]
    pet_category = target_variables.iloc[:,-1]
    print(f"shape of breed_category:{breed_category.shape}")
    print(f"shape of pet_category:{pet_category.shape}")

    #results_df_breed = best_estimator_rf(X,breed_category)
    #print(results_df_breed)
    #results_df_pet_category = best_estimator_rf(X,pet_category)
    #print(results_df_pet_category.head())







# =============================================================================
#     logreg_params = {}
#     logreg_params["C"] =  [0.01, 0.1, 10, 100]
#     logreg_params["fit_intercept"] =  [True, False]
#     logreg_params["warm_start"] = [True,False]
#     logreg_params["random_state"] = [1]
#
#     lr_dist = {}
#     lr_dist["C"] = stats.expon(scale=.01)
#     lr_dist["fit_intercept"] =  [True, False]
#     lr_dist["warm_start"] = [True,False]
#     lr_dist["random_state"] = [1]
# =============================================================================

    # logregression_grid = hyperParameter_Lr(prepocessed_train_updated,breed_category)

    #Knn_hyperParameter(x,breed_category)
    #Knn_hyperParameter(x,pet_category)
    # hyperParameter_Lr(prepocessed_train_updated,breed_category)
    # results_dt_df = DecisionTreeHyperParameter(prepocessed_train_updated,breed_category)
    # results_dt_df_pet = DecisionTreeHyperParameter(prepocessed_train_updated,pet_category)

    GradientBoosting(prepocessed_train_updated,breed_category)