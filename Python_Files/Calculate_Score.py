# -*- coding: utf-8 -*-
"""
Created on Fri Dec  2 23:50:36 2022

@author: semih
"""

# Kütüphaneler
from sklearn.model_selection import cross_val_score
from sklearn.metrics import r2_score
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
import math
# -----------------------------------------------------------------------------


def Calculate(X_train, X_test, y_train, y_test, algorithm, algorithm_pred):
    return_values = {}
    mse = mean_squared_error(y_test, algorithm_pred)
    rmse = math.sqrt(mse)
    mae = mean_absolute_error(y_test, algorithm_pred)
    cross_val = cross_val_score(algorithm, X_train, y_train, cv=10)
    r2 = r2_score(y_test, algorithm_pred)
    return_values = {'Mean Squared Error'       :   mse,
                     'Root Mean Squared Error'  :   rmse,
                     'Mean Absolute Error'      :   mae,
                     'Cross Validation'         :   cross_val,
                     'R2 Score'                 :   r2}
    
    return return_values


def Remainder_Calculate(predict_value, real_value):
    return predict_value - real_value