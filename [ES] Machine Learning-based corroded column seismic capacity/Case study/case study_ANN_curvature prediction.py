#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
import os
import shutil
import time

starttime = time.time()

t_array = [0,25,50,75,100]
ti = 0
for t_current in t_array:
    ti = ti + 1
    X_train_path = '../Training_InputData.txt'
    y_train_path = '../Training_OutputData.txt'
    X_train = pd.read_csv(X_train_path, sep='\t')
    y_train = pd.read_csv(y_train_path, sep='\t')
    X_test_path = 'case study_database/'+str(t_current)+'year/InputData.txt'
    y_test_path = 'case study_database/'+str(t_current)+'year/OutputData.txt'
    X_test = pd.read_csv(X_test_path, sep='\t')
    y_test = pd.read_csv(y_test_path, sep='\t')

    # Create folder to save result files
    if ti == 1:
        if os.path.exists('results'):
            shutil.rmtree('results')
    os.makedirs('results/' + str(t_current) + 'year')

    # Neural Network
    from h2o.estimators import H2ODeepLearningEstimator
    bestN = 1
    bestQ = 8

    model_nn = H2ODeepLearningEstimator(distribution="gaussian",
                                        loss="Quadratic",
                                        hidden=[bestQ] * bestN,
                                        activation="tanh",
                                        epochs=10000,
                                        variable_importances=True,
                                        standardize=True
                                        )

    import numpy as np
    import matplotlib.pyplot as plt
    from sklearn.metrics import make_scorer, r2_score, mean_absolute_error, mean_squared_error

    def model_fit(estimator, idx, X_train, X_test, y_train, y_test):
        # Fit model
        if hasattr(estimator, 'train'):
            x = X_train.columns.values.tolist()
            y = y_train.columns[idx]
            df = X_train.copy()
            df[y] = y_train[y]
            df = h2o.H2OFrame(df)
            df_test = X_test.copy()
            df_test[y] = y_test[y]
            df_test = h2o.H2OFrame(df_test)
            estimator.train(training_frame=df, x=x, y=y)
            train_pred = estimator.predict(df).as_data_frame().values
            test_pred = estimator.predict(df_test).as_data_frame().values
        else:
            estimator.fit(X_train, y_train.iloc[:,idx])
            train_pred = estimator.predict(X_train)
            test_pred = estimator.predict(X_test)
        y_train_pred = np.squeeze(train_pred)
        y_test_pred = np.squeeze(test_pred)
        # Return performance
        train_MSE = mean_squared_error(y_train.iloc[:,idx], y_train_pred)
        test_MSE = mean_squared_error(y_test.iloc[:,idx], y_test_pred)
        return y_train_pred, y_test_pred, train_MSE, test_MSE


    import h2o
    h2o.init()
    current_model = model_nn
    model_name = "Artificial Neural Network"
    y_which_col = 1  # [0]phi_1, [1]phi_2, [2]phi_3, [3]phi_4
    s_name = y_train.columns[y_which_col]  #
    # final model (results of 100 runs)
    for i in range(1,101):
        print('Year',t_current,', No.',i,'run: ')
        y_train_pred, y_test_pred, train_MSE, test_MSE = model_fit(current_model, y_which_col, X_train, X_test, y_train, y_test)
        other_results = []
        other_results.append([model_name, train_MSE, test_MSE])
        y_train_pred_ori = y_train_pred
        y_train_true_ori = y_train.iloc[:,y_which_col].values
        y_test_pred_ori = y_test_pred
        y_test_true_ori = y_test.iloc[:,y_which_col].values
        pd.DataFrame.from_dict({"y_train_pred": y_train_pred_ori, "y_train_true": y_train_true_ori}).to_excel("results/" + str(t_current) + "year/prediction" + "_" + s_name + "_" + model_name + "_i" + str(i) + "_ytrain_ori.xlsx")
        pd.DataFrame.from_dict({"y_test_pred": y_test_pred_ori, "y_test_true": y_test_true_ori}).to_excel("results/" + str(t_current) + "year/prediction" + "_" + s_name + "_" + model_name + "_i" + str(i) + "_ytest_ori.xlsx")
        pd.DataFrame(other_results, columns=["Model Name", "train_MSE", "test_MSE"]).to_excel("results/" + str(t_current) + "year/prediction" + "_" + s_name + "_i" + str(i) + "_other_results.xlsx")

print('All done! ^_^=====================^_^=====================^_^')

endtime = time.time()
runtime = endtime - starttime
print('\n============================================')
print('RunTime =',runtime/60,'Minutes')