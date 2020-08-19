# -*- coding: utf-8 -*-
"""
Created on Wed Aug  5 21:51:18 2020

@author: 1094356
"""

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import KNNImputer


def baisc_details(dataset):
    print(f"shape:{dataset.shape}")
    print('-'*50)
    print(dataset.columns)
    print('-'*50)
    print(dataset.head())
    print('-'*50)
    print("=====datatypes of each columns========")
    print(dataset.dtypes)
    

def convert_toDateTime(dataset,column):
    dataset[column] = dataset[column].apply(pd.to_datetime)
    dataset['total_days'] = (dataset[column[1]] - dataset[column[0]]).dt.days/365
    dataset['total_days'] =  dataset['total_days'].apply(lambda x : "{:.2f}".format(x))
    return dataset
def check_missing_values(dataset):
    print("---checking missing values-------")
    print(dataset.isnull().sum())   
    print('only condition has missing values')
    
def values_count(dataset,column):
    print(f"values of each variable:\n{dataset[column].value_counts()}")
    dict_count = dataset[column].value_counts().to_dict()
    return dict_count
    
    
    
    
def fill_missing_values(dataset):
    dataset.fillna(float(3),inplace = True)
    
    return dataset
def KNN_imputer(dataset,column):
    imputer = KNNImputer(n_neighbors=2)
    imputed_value = imputer.fit_transform(dataset[column].values.reshape(-1,1))
    #imputed_value = imputer.fit_transform(dataset[column].values.reshape(-1,1))
    dataset[column] = imputed_value
    return dataset

def drop_columns(dataset):
    print(f"'pet_id', 'issue_date', 'listing_date' columns has been dropped")
    dataset.drop(['pet_id', 'issue_date', 'listing_date'], axis = 1,inplace = True)
    
    return dataset
    
def missing_As_0(dataset,column,replace_value = None):
    dataset[column] = dataset[column].apply(lambda x: replace_value if x == 0 else x )
    return dataset 
    



def variable_scaling(dataset,column):
    dataset[column] = dataset[column].apply(lambda x: x*100)
    return dataset

def lableEncoding(dataset,column):
    label_encoder = LabelEncoder()
    encoded_values =   label_encoder.fit_transform(dataset[column].values)
    print('replace the value in dataset')
    dataset[column] = encoded_values
    
    return dataset

def convet_type(dataset,column):
    dataset[column] =dataset[column].apply(lambda x : int(x))
    
    return dataset


def conversion(val):
    if val == 0:
        return 0
    elif val == 13:
        return 1
    else:
        return 2

def x1_column(dataset,column):
    dataset[column] =dataset[column].apply(lambda x : conversion(x))
    return dataset

def target_values(dataset,column):
    dataset[column] =dataset[column].apply(lambda x : int(3) if x == int(4) else int(x))
    return dataset
     

def replaceLessCount(dataset,column,count_dict):
    dataset[column] = dataset[column].apply(lambda x : 'others' if count_dict[x] < 10 else x)
    
    colors_name = []
    for key, val in count_dict.items():
        if count_dict[key]<10:
            colors_name.append(key)
            
    return dataset,colors_name

if __name__ == '__main__':
    print('started working')
    train = pd.read_csv('./Dataset/train.csv')
    test = pd.read_csv('./Dataset/test.csv')
    
    print("="*20 +'printing basic details' +"="*20)
    baisc_details(train)
    
    print("="*20 +'checking missing values' +"="*20)
    check_missing_values(train)
    
    print('--- count the values---')
    values_count(train,'condition')
    
    print(' vale 2 is less in number so we will fill 2 inplace of nan')
    print(' adding one more condition 3 ')
    print("="*20 +'fill the  missing values for condition column' +"="*20)
    dataset_1 = fill_missing_values(train)
    #dataset_1 = KNN_imputer(train,'condition')
    print()
    print('---------now check value counts again------')
    values_count(dataset_1,'condition')
    
    dataset_2 = convert_toDateTime(dataset_1,['issue_date', 'listing_date'])
    print('basic details after date time conversion')
    baisc_details(dataset_2)
    
    
    print("="*10 +'since our task is to predict bread category and pet category, some columns are not useful ' +"="*10)
    dataset_2 = drop_columns(dataset_2)
    print()
    print('-- basic detail after droping some columns---')
    baisc_details(dataset_2)
    print(dataset_2.head())
    
    print("===== starting univariate analysis============")
    
    print('length is 0 at many place which is invalid , based on observation replace it with 1.01')
    dataset_3 = missing_As_0(dataset_2,'length(m)',replace_value = 1.01)
    print('length in (m) and height in (cm) convert thme is same scale')
    
    dataset_3 = variable_scaling(dataset_3,'length(m)')
    print('checking the value after scaling')
    print(dataset_3['length(m)'].head())
    
    color_count = values_count(dataset_3,'color_type')
    dataset_4,color_list = replaceLessCount(dataset_3,'color_type',color_count)
    print(" color_type is as string but fit only works for numeric variable")
    dataset_4 = lableEncoding(dataset_4,'color_type')
    color_count_new = values_count(dataset_4,'color_type')
    #print(dataset_4['color_type'].head())
    print('later we have have to do one hot encoding which takes only numerical values')
    
    print('convert the condition columns to integer')
    dataset_5= convet_type(dataset_4,'condition')
    print(dataset_5['condition'].head())
    
    
    print('here we have two target variable')
    print('conver breed_category to int')
    dataset_6 = convet_type(dataset_5,'breed_category')
    print(dataset_6['breed_category'].head())
    
    
    
    print(' count te value of each traget variable')
    for target in ['breed_category','pet_category']:
        values_count(dataset_6,target)
        print("="*100)
        
    #print(' i clud see that number 3 is missing as of now i ma replacing 4 with 3 later will do somting')
    
    #dataset_7 = target_values(dataset_6,'pet_category')
    
    #print('checkng value clunts again')
    #values_count(dataset_7,'pet_category')
    
    print('x1 is right skewed keep 0 as it is conver 13 to 1 rest 2')
    
    #dataset_7['X1'] = np.log10( dataset_7['X1'])
    dataset_7 = x1_column(dataset_6,'X1')
    
    
    print('now we can save the prepocessed traing dataset')
    dataset_7.to_csv('./Dataset/prepocessed_train.csv',index = False)