# -*- coding: utf-8 -*-
"""
Created on Thu Aug  6 20:47:47 2020

@author: 1094356
"""
import pandas as pd
import DatapreProcessing as dp
import model_training as mt




def replace_color(dataset,column,color_list):
    dataset[column] =  dataset[column].apply(lambda x : 'other' if x in color_list else x )
    return dataset

if __name__== '__main__':
    print('test data preprocessing')
    test = pd.read_csv('./Dataset/test.csv')
    dp.baisc_details(test)
    dp.values_count(test, 'color_type')
    submission = pd.DataFrame(test.pet_id, columns=['pet_id'])
    submission.to_csv('./Dataset/submission.csv',index = False)
    print(submission.head())
    dp.check_missing_values(test)
      
    test_1 = dp.fill_missing_values(test)
    print(test_1.head())
    dp.check_missing_values(test_1)
    
    test_1 = dp.convert_toDateTime(test_1,['issue_date', 'listing_date'])
    print('basic details after date time conversion')
    dp.baisc_details(test_1)
    
    test_2 = dp.drop_columns(test_1)
    print(test_2.head())
    
    print('length is 0 at many place which is invalid , based on observation replace it with 1.01')
    test_3 = dp.missing_As_0(test_2,'length(m)',replace_value = 1.01)
    test_3 = dp.variable_scaling(test_3,'length(m)')
    print(test_3.head())
    
    colors_name = ['Chocolate Point', 'Pink', 'Green', 'Blue Smoke', 'Silver Lynx Point', 'Agouti', 'Brown Tiger', 'Liver', 'Black Tiger', 'Liver Tick']
    #_,colors_name = dp.values_count(test_3,'color_type')
    test_4 = replace_color(test_3,'color_type',colors_name)
    #_,colors_name = dp.replaceLessCount(test_3,'color_type',colors_name)
    print(" color_type is as string but fit only works for numeric variable")
    color_count_test_updated = dp.values_count(test_4,'color_type')
    
    
    test_4 = dp.lableEncoding(test_3,'color_type')
    print(test_4.head())
    test_4 = dp.x1_column(test_4,'X1')
    
    color_type_encoded = mt.oneHotEncoder(test_4,'color_type')
    test_4 = pd.concat([test_4, color_type_encoded ] , axis = 1)
    #print(X1.head())
    test_4.drop('color_type',axis = 1 ,inplace = True)
    
    print("==condition== value counts")
    dp.values_count(test_4,'condition')
    test_4.to_csv('./Dataset/test_updated.csv',index = False)