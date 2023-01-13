# -*- coding: utf-8 -*-
"""
Created on Sun Dec  4 13:49:34 2022

@author: semih
"""

# Kütüphaneler
from tensorflow.keras import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.losses import MeanSquaredLogarithmicError
import Calculate_Score as cs
from sklearn.metrics import r2_score
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import math
import pandas as pd
# -----------------------------------------------------------------------------

def Nural_Network_Semih(X_train, X_test, y_train, y_test):
    # Creating model using the Sequential in tensorflow
   
    hidden_units1 = 160
    hidden_units2 = 480
    hidden_units3 = 256
    learning_rate = 0.01
    def build_model_using_sequential():
        model = Sequential([
        Dense(hidden_units1, kernel_initializer='normal', activation='relu'),
        Dropout(0.2),
        Dense(hidden_units2, kernel_initializer='normal', activation='relu'),
        Dropout(0.2),
        Dense(hidden_units3, kernel_initializer='normal', activation='relu'),
        Dense(1, kernel_initializer='normal', activation='linear')
      ])
        return model
    # build the model
    model = build_model_using_sequential()
    # loss function
    msle = MeanSquaredLogarithmicError()
    model.compile(
        loss=msle, 
        optimizer=Adam(learning_rate=learning_rate), 
        metrics=[msle]
    )
    # train the model
    history = model.fit(
        X_train.values, 
        y_train.values, 
        epochs=10, 
        batch_size=64,
        validation_split=0.2
    )
    def plot_history(history, key):
      plt.plot(history.history[key])
      plt.plot(history.history['val_'+key])
      plt.xlabel("Epochs")
      plt.ylabel(key)
      plt.legend([key, 'val_'+key])
      plt.show()
    # Plot the history
    plot_history(history, 'mean_squared_logarithmic_error')
    
    neural_preds = model.predict(X_test)
    neural_preds_ed = []
    for i in neural_preds:
        for x in i:
            neural_preds_ed.append(x)
    neural_preds_ed2 = pd.Series(neural_preds_ed)
    #neural_preds = pd.Series(neural_preds)

    neural_df = pd.DataFrame({'PREDICT' : neural_preds_ed, 'TEST' : y_test})
    neural_df['REMAINDER'] = neural_df.apply(lambda x: cs.Remainder_Calculate(x.PREDICT, x.TEST), axis=1)
    neural_df['REMAINDER_ABS'] = abs(neural_df['REMAINDER'])


    
    neural_mse = mean_squared_error(y_test, neural_preds_ed)
    neural_rmse = math.sqrt(neural_mse)
    neural_mae = mean_absolute_error(y_test, neural_preds_ed)
    neural_r2 = r2_score(y_test, neural_preds_ed)
    
    
    neural_values = {'Neural Network Regression Preds'          : neural_preds_ed2,
                     'Neural Network Mean Squared Error'        : neural_mse,
                     'Neural Network Root Mean Squared Error'   : neural_rmse,
                     'Neural Network Mean Absolute Error'       : neural_mae,
                     'Neural Network R2 Score'                  : neural_r2,
                     'Neural Network Stats DF'                  : neural_df}
    
    
    
    return neural_values
    
    
    
    
    
    