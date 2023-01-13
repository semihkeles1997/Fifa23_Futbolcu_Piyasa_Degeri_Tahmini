# -*- coding: utf-8 -*-
"""
Created on Fri Dec  2 23:42:53 2022

@author: semih
"""

# Kütüphaneler
from sklearn.neighbors import KNeighborsRegressor
import Calculate_Score as cs
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV
# -----------------------------------------------------------------------------

def KNN_Algorithm_Semih(X_train, X_test, y_train, y_test):
    # KNN
    # En iyi K değerinin belirlenmesi
    params = {'n_neighbors' : range(50)}
    knn = KNeighborsRegressor()
    model = GridSearchCV(knn, params, cv=10)
    model.fit(X_train,y_train)
    print("BEST: ",model.best_params_)

    # Belirlenen en iyi K değeri için KNN algoritmasının oluşturulması
    best_knn = KNeighborsRegressor(n_neighbors=model.best_params_['n_neighbors'])
    #best_knn = KNeighborsRegressor(n_neighbors=5)
    best_knn.fit(X_train, y_train)
    best_knn_pred = best_knn.predict(X_test)
    best_knn_pred_abs = abs(best_knn_pred)
    
    
    best_knn_calculated_score = cs.Calculate(X_train, X_test, y_train, y_test, best_knn, best_knn_pred)
    
    best_knn_score_train = best_knn.score(X_train, y_train)
    best_knn_score_test = best_knn.score(X_test, y_test)

    best_knn_stats_df = pd.DataFrame({'PREDICT' : best_knn_pred, 'TEST' : y_test})
    best_knn_stats_df['REMAINDER'] = best_knn_stats_df.apply(lambda x: cs.Remainder_Calculate(x.PREDICT, x.TEST), axis=1)
    best_knn_stats_df['REMAINDER_ABS'] = abs(best_knn_stats_df['REMAINDER'])

    best_knn_values = {'Best KNN Preds'         : best_knn_pred,
                    'Best KNN Train Score'      : best_knn_score_train,
                    'Best KNN Test Score'       : best_knn_score_test,
                    'Best KNN Calculated Score' : best_knn_calculated_score,
                    'Best KNN Stats DF'         : best_knn_stats_df}

    # Algoritmanın çizilmesi
    plt.figure(figsize=(10,6))
    plt.plot(y_test,best_knn_pred_abs,'ro',color='red', markerfacecolor='red',markersize='1')
    plt.title("KNN Tahmin Değerleri ile Gerçek Değerler Arasındaki İlişki")
    plt.xlabel("Tahmin Değerleri")
    plt.ylabel("Gerçek Değerler")
    plt.show()
    
    
    return best_knn_values
    
    
    
    
    
    
    
    
    
    
    
    
    
    