# -*- coding: utf-8 -*-
"""
Created on Sat Dec  3 00:18:29 2022

@author: semih
"""

# Kütüphaneler
import Calculate_Score as cs
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
# -----------------------------------------------------------------------------

def Linear_Regression_Semih(X_train, X_test, y_train, y_test):
    # Linear Regression
    reg_main = LinearRegression().fit(X=X_train, y=y_train)
    reg_main_coef = reg_main.coef_
    reg_main_intercept = reg_main.intercept_
    reg_main_preds = reg_main.predict(X_test)
    reg_main_pred_abs = abs(reg_main_preds)
    reg_main_train_score = reg_main.score(X_train, y_train)
    reg_main_test_score = reg_main.score(X_test, y_test)
    
    reg_stats_df = pd.DataFrame({'PREDICT' : reg_main_preds, 'TEST' : y_test})
    reg_stats_df['REMAINDER'] = reg_stats_df.apply(lambda x: cs.Remainder_Calculate(x.PREDICT, x.TEST), axis=1)
    reg_stats_df['REMAINDER_ABS'] = abs(reg_stats_df['REMAINDER'])
    
    lr_calculated_score = cs.Calculate(X_train, X_test, y_train, y_test, reg_main, reg_main_preds)
    
    lr_values = {'Linear Regression Preds'              : reg_main_preds,
                 'Linear Regression Train Score'        : reg_main_train_score,
                 'Linear Regression Test Score'         : reg_main_test_score,
                 'Linear Regression Calculated Scores'  : lr_calculated_score,
                 'Linear Regression Stats DF'           : reg_stats_df}
    
    plt.figure(figsize=(10,6))
    plt.plot(y_test,reg_main_pred_abs,'ro',color='blue', markerfacecolor='red',markersize='1')
    plt.title("Linear Regression Tahmin Değerleri ile Gerçek Değerler Arasındaki İlişki")
    plt.xlabel("Gerçek Değerler")
    plt.ylabel("Tahmin DeğerleriR")
    plt.show()
   
    
    Y_max = y_test.max()
    Y_min = y_test.min()
    import seaborn as sns
    import numpy as np
    plt.title("Linear Regression Gerçek Değerler ile Tahmin Değerleri Arasındaki İlişki")
    ax = sns.scatterplot(x=y_test, y=reg_main_preds)
    ax.set(ylim=(Y_min, Y_max))
    ax.set(xlim=(Y_min, Y_max))
    ax.set_xlabel("Gerçek Değerler")
    ax.set_ylabel("Tahmin Değerleri")
    
    X_ref = Y_ref = np.linspace(Y_min, Y_max, 100)
    plt.plot(X_ref, Y_ref, color='red', linewidth=1)
    plt.show()
    
   
    return lr_values
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    