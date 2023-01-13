# -*- coding: utf-8 -*-
"""
Created on Sat Dec  3 17:13:00 2022

@author: semih
"""

# Kütüphaneler
from sklearn.svm import SVR
import Calculate_Score as cs
import pandas as pd
import matplotlib.pyplot as plt
# -----------------------------------------------------------------------------

def Support_Vector_Machine_Semih(X_train, X_test, y_train, y_test):
    
    svr_regressor = SVR(kernel='rbf') # default rbf
    svr_regressor.fit(X_train, y_train)
    svr_preds = svr_regressor.predict(X_test)
    svr_preds_abs = abs(svr_preds)
    svr_train_score = svr_regressor.score(X_train, y_train)
    svr_test_score = svr_regressor.score(X_test, y_test)
    
    svr_stats_df = pd.DataFrame({'PREDICT' : svr_preds, 'TEST' : y_test})
    svr_stats_df['REMAINDER'] = svr_stats_df.apply(lambda x: cs.Remainder_Calculate(x.PREDICT, x.TEST), axis=1)
    svr_stats_df['REMAINDER_ABS'] = abs(svr_stats_df['REMAINDER'])
    
    
    svr_calculated_score = cs.Calculate(X_train, X_test, y_train, y_test, svr_regressor, svr_preds)
    
    
    svr_values = {'Support Vector Machine Preds'             : svr_preds,
                 'Support Vector Machine Train Score'        : svr_train_score,
                 'Support Vector Machine Test Score'         : svr_test_score,
                 'Support Vector Machine Calculated Scores'  : svr_calculated_score,
                 'Support Vector Machine Stats DF'           : svr_stats_df}
    
    
    plt.figure(figsize=(10,6))
    plt.plot(y_test,svr_preds_abs,'ro',color='blue', markerfacecolor='red',markersize='1')
    plt.title("Support Vector Machine Tahmin Değerleri ile Gerçek Değerler Arasındaki İlişki")
    plt.xlabel("Gerçek Değerler")
    plt.ylabel("Tahmin DeğerleriR")
    plt.show()
    
    return svr_values
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    