# -*- coding: utf-8 -*-
"""
Created on Sat Dec  3 00:05:10 2022

@author: semih
"""

# Kütüphaneler
import Calculate_Score as cs
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import tree
from sklearn.tree import DecisionTreeRegressor
# -----------------------------------------------------------------------------

def Decision_Tree_Semih(X_train, X_test, y_train, y_test):
    # Decision Tree
    regressor = DecisionTreeRegressor(random_state=0, max_depth=5)
    regressor.fit(X_train, y_train)
    dtree_pred = regressor.predict(X_test)
    
    dtree_pred_abs = abs(dtree_pred)
    dtree_stats_df = pd.DataFrame({'PREDICT' : dtree_pred, 'TEST' : y_test})
    dtree_stats_df['REMAINDER'] = dtree_stats_df.apply(lambda x: cs.Remainder_Calculate(x.PREDICT, x.TEST), axis=1)
    dtree_stats_df['REMAINDER_ABS'] = abs(dtree_stats_df['REMAINDER'])
    dtree_stats_df['REMAINDER_ABS'] = abs(dtree_stats_df['REMAINDER'])
    dtree_score_train = regressor.score(X_train, y_train)
    dtree_score_test = regressor.score(X_test, y_test)


    dtree_calculated_score = cs.Calculate(X_train, X_test, y_train, y_test, regressor, dtree_pred)

    dtree_values = {'Decision Tree Preds'               : dtree_pred,
                    'Decision Tree Train Score'         : dtree_score_train,
                    'Decision Tree Test Score'          : dtree_score_test,
                    'Decision Tree Calculated Scores'   : dtree_calculated_score,
                    'Decision Tree Stats DF'            : dtree_stats_df}


    fig = plt.figure(figsize=(100,20))
    _ = tree.plot_tree(regressor, 
                       feature_names=X_train.columns,
                       filled=True)
    fig.savefig("decistion_tree.png")




    return dtree_values

































