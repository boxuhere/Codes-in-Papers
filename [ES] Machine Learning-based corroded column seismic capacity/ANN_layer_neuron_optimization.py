#!/usr/bin/env python
# coding: utf-8

import os
import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error

starttime = time.time()

x_path = 'Training_InputData.txt'
y_path = 'Training_OutputData.txt'
X = pd.read_csv(x_path, sep='\t')
Y = pd.read_csv(y_path, sep='\t')

n_fold = 5
samples_per_fold = (len(X_train)//n_fold)+1
training_idx = X_train.index.values.tolist()
np.random.shuffle(training_idx)
train_masks = []
for i in range(n_fold):
    if i != n_fold-1:
        train_masks.append(training_idx[i*samples_per_fold:(i+1)*samples_per_fold])
    else:
        train_masks.append(training_idx[i*samples_per_fold:])

X_train = X.loc[training_idx, :]
y_train = Y.loc[training_idx, :]
X_test = X.loc[list(set(X.index.values.tolist())-set(training_idx)),:]
y_test = Y.loc[list(set(X.index.values.tolist())-set(training_idx)),:]


def cv_fit(estimator, params, i_fold, idx, X_train, X_eval, y_train, y_eval):    
    results = []
    # Fit model
    for i in idx:
        if hasattr(estimator, 'train'):
            x = X_train.columns.values.tolist()
            y = y_train.columns[i]
            df = X_train.copy()
            df[y] = y_train[y]
            df_valid = X_eval.copy()
            df_valid[y] = y_eval[y]
            df = h2o.H2OFrame(df)
            df_valid = h2o.H2OFrame(df_valid)
            estimator.train(training_frame=df, validation_frame=df_valid, x=x, y=y)
            train_pred = estimator.predict(df).as_data_frame().values
            eval_pred = estimator.predict(df_valid).as_data_frame().values
        else:
            estimator.fit(X_train, y_train.iloc[:,i])
            train_pred = estimator.predict(X_train)
            eval_pred = estimator.predict(X_eval)
            
        y_train_pred = np.squeeze(train_pred)
        y_eval_pred = np.squeeze(eval_pred)
        
        # Return performance
        train_MSE = mean_squared_error(y_train.iloc[:,i], y_train_pred)
        eval_MSE = mean_squared_error(y_eval.iloc[:,i], y_eval_pred)
        if hasattr(estimator, 'coef_'):
            coefs = estimator.coef_
        else:
            coefs = pd.DataFrame(estimator.varimp()).set_index(0).loc[X.columns.values.tolist()][1].values
        n_zeros = sum(coefs==0)
        results.append([i_fold, i, *params, train_MSE, eval_MSE, n_zeros])
    return results


import h2o
h2o.init()
y_col = 3;  # [0]phi_1, [1]phi_2, [2]phi_3, [3]phi_4
y_train.columns[y_col]

# Neural Network
from h2o.estimators import H2ODeepLearningEstimator
writer = pd.ExcelWriter("./results/NN_cv.xlsx")
for j in range(1,5,1):
    nn_results = []
    for n_layer in range(1,6,1):
        for n_neuron in range(1,11,1):
            for i in range(n_fold):
                print('--Runing loop:',j,',n_layer:',n_layer,',n_neuron:',n_neuron,',n_fold:',i)
                model_nn = H2ODeepLearningEstimator(distribution="gaussian",
                                     loss="Quadratic",
                                     hidden=[n_neuron]*n_layer,
                                     activation = "tanh",
                                     epochs=10000,
                                     variable_importances=True,
                                     standardize=True
                                     )
                train_idx = list(set(training_idx)-set(train_masks[i]))
                eval_idx = list(set(train_masks[i]))
                np.random.shuffle(train_idx)  # random sort
                np.random.shuffle(eval_idx)
                X_train_cv = X_train.loc[train_idx,:]
                X_eval_cv = X_train.loc[eval_idx, :]
                y_train_cv = y_train.loc[train_idx, :]
                y_eval_cv = y_train.loc[eval_idx, :]
                result_ = cv_fit(model_nn, [n_layer, n_neuron], i, [y_col], X_train_cv, X_eval_cv, y_train_cv, y_eval_cv)
                nn_results.extend(result_)
    nn_results_save = pd.DataFrame(nn_results, columns=["i_fold", "y_i", "n_layer","n_neuron", "training_MSE", "validation_MSE", "n_zero_coefs"]).set_index(["y_i"]) # 应该是"validation_MSE" ??
    nn_results_save.to_excel(writer, sheet_name = 'loop'+str(j))
writer.save()
print('--Training done!')            

endtime = time.time()
runtime = endtime - starttime
print('\n============================================')
print('RunTime =',runtime/60,'Minutes')