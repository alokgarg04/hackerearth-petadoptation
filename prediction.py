# -*- coding: utf-8 -*-
"""
Created on Thu Aug  6 20:36:03 2020

@author: 1094356
"""
import pandas as pd
import numpy as np
import joblib
import DatapreProcessing as dp

def predict(model,test_data):
    predicted = model.predict(test_data)
    return predicted




if __name__ == '__main__':
    print('prediction started')
    test = pd.read_csv('./Dataset/test.csv')
    test_updated = pd.read_csv('./Dataset/test_updated.csv')
    submission = pd.read_csv('./Dataset/submission.csv')

    print('load the trained models')
    breed_category = joblib.load('./models/breed_category_17th.pkl')
    pet_category =  joblib.load('./models/pet_category_17th.pkl')

    submission['breed_category'] = predict(breed_category,test_updated)
    submission['pet_category'] =  predict(pet_category,test_updated)
    submission.to_csv('./Dataset/submission_17th.csv',index = False)
    #dp.baisc_details(test)