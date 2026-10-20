"""
Traditional machine learning algorithms for time series classification.

This module contains implementations of traditional machine learning
algorithms optimized for time series classification tasks, including
hyperparameter optimization via Optuna.
"""

import pandas as pd
import numpy as np
import time
import optuna
import torch
from datetime import datetime

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix,classification_report
from sklearn.preprocessing import StandardScaler, OneHotEncoder

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
import xgboost as xgb
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.neural_network import MLPClassifier

import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

def optimize_LOGRG(x_train, x_val, x_test, y_train, y_val, y_test, namee):
    max_iter=10000

    def objective_LOGRG(trial):
        n_jobs= -1
        c = trial.suggest_float('C', 1e-10, 1000, log=True)
        #class_weight = trial.suggest_categorical('class_weight', ['balanced', None])
        fit_intercept = trial.suggest_categorical('fit_intercept', [True, False])
        solver = trial.suggest_categorical('solver', ['saga'])
        penalty = trial.suggest_categorical('penalty', ['elasticnet'])
        l1_ratio = trial.suggest_float('l1_ratio', 0, 1)

        model = LogisticRegression( max_iter=max_iter, solver=solver, penalty=penalty, l1_ratio=l1_ratio,C=c,n_jobs=n_jobs, random_state=42, fit_intercept=fit_intercept)
    
        model.fit(x_train, y_train)
        y_pred = model.predict(x_val)
        score = accuracy_score(y_val, y_pred)

        return score
    
    study = optuna.create_study(direction='maximize')
    study.optimize(objective_LOGRG, n_trials=100)
    start=time.time()
    best_trial = study.best_trial
    best_params = study.best_params
    best_score = best_trial.value
    best_model = LogisticRegression(**best_params, random_state=42)
    best_model.fit(x_train, y_train)
    y_pred = best_model.predict(x_test)
    test_score = accuracy_score(y_test, y_pred)
    end=time.time()
    print(end-start, " seconds")
    classification_rep = classification_report(y_test, y_pred)

    current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    filename = f"{namee}_classification_report.txt"
    with open(filename, 'w') as file:
        file.write(classification_rep)

    print(f"Classification report saved as {filename}")

    return best_params, test_score



def optimize_DT(x_train, x_val, x_test, y_train, y_val, y_test,namee):
    #just import libraries
    def objective_DT(trial):
        #n_estimators = trial.suggest_int('n_estimators', 10, 200)
        max_depth = trial.suggest_int('max_depth', 2, 32)
        criterion=trial.suggest_categorical('criterion', ['gini', 'entropy', 'log_loss'])
        model = DecisionTreeClassifier(
            max_depth=max_depth
            ,criterion=criterion
        )
        #x_train2, x_val, y_train2, y_val = train_test_split(x_train, y_train, test_size=0.25)
        model.fit(x_train, y_train)
        y_pred = model.predict(x_val)
        score = accuracy_score(y_val, y_pred)

        return score
    
    study = optuna.create_study(direction='maximize')
    study.optimize(objective_DT, n_trials=100)
    start=time.time()
    best_trial = study.best_trial
    best_params = best_trial.params
    best_score = best_trial.value

    best_model = DecisionTreeClassifier(**best_params, random_state=42)
    best_model.fit(x_train, y_train)
    y_pred = best_model.predict(x_test)
    test_score = accuracy_score(y_test, y_pred)
    end=time.time()
    print(end-start, " seconds")
    classification_rep = classification_report(y_test, y_pred)
    current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    filename = f"{namee}_classification_report.txt"
    with open(filename, 'w') as file:
        file.write(classification_rep)

    print(f"Classification report saved as {filename}")

    return best_params, test_score

def optimize_RF(x_train, x_val, x_test, y_train, y_val, y_test,namee):
    #just import libraries
    def objective_RF(trial):
        n_estimators = trial.suggest_int('n_estimators', 10, 200)
        max_depth = trial.suggest_int('max_depth', 2, 32)

        model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth
        )
        #x_train2, x_val, y_train2, y_val = train_test_split(x_train, y_train, test_size=0.25)
        model.fit(x_train, y_train)
        y_pred = model.predict(x_val)
        score = accuracy_score(y_val, y_pred)

        return score

    study = optuna.create_study(direction='maximize')
    study.optimize(objective_RF, n_trials=100)
    start=time.time()
    best_trial = study.best_trial
    best_params = study.best_params
    best_score = best_trial.value
    best_model = RandomForestClassifier(**best_params, random_state=42)
    best_model.fit(x_train, y_train)
    y_pred = best_model.predict(x_test)
    test_score = accuracy_score(y_test, y_pred)
    end=time.time()
    print(end-start, " seconds")
    classification_rep = classification_report(y_test, y_pred)
    current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    filename = f"{namee}_classification_report.txt"
    with open(filename, 'w') as file:
        file.write(classification_rep)

    print(f"Classification report saved as {filename}")

    return best_params, test_score

def optimize_KNN(x_train, x_val, x_test, y_train, y_val, y_test,namee):
    #just import libraries
    def objective_KNN(trial):
        n_neighbors = trial.suggest_int('n_neighbors', 1, 20)
        weights = trial.suggest_categorical('weights', ['uniform', 'distance'])
        algorithm = trial.suggest_categorical('algorithm', ['auto', 'ball_tree', 'kd_tree', 'brute'])

        model = KNeighborsClassifier(n_neighbors=n_neighbors, weights=weights, algorithm=algorithm)

        #x_train2, x_val, y_train2, y_val = train_test_split(x_train, y_train, test_size=0.25)
        model.fit(x_train, y_train)
        y_pred = model.predict(x_val)
        score = accuracy_score(y_val, y_pred)

        return score

    study = optuna.create_study(direction='maximize')
    study.optimize(objective_KNN, n_trials=100)
    start=time.time()
    best_trial = study.best_trial
    best_params = study.best_params
    best_score = best_trial.value
    best_model = KNeighborsClassifier(**best_params)
    best_model.fit(x_train, y_train)
    y_pred = best_model.predict(x_test)
    test_score = accuracy_score(y_test, y_pred)
    end=time.time()
    print(end-start, " seconds")
    classification_rep = classification_report(y_test, y_pred)
    current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    filename = f"{namee}_classification_report.txt"
    with open(filename, 'w') as file:
        file.write(classification_rep)

    print(f"Classification report saved as {filename}")

    return best_params, test_score

def optimize_XGBOOST(x_train, x_val, x_test, y_train, y_val, y_test,namee):
    #import libraries
    def objective_XGBOOST(trial):
        n_estimators = trial.suggest_int('n_estimators', 10, 200)
        learning_rate = trial.suggest_float('learning_rate', 0.001, 1.0)
        max_depth = trial.suggest_int('max_depth', 2, 32)
        min_child_weight = trial.suggest_float('min_child_weight', 1, 10)
        subsample = trial.suggest_float('subsample', 0.1, 1)
        colsample_bytree = trial.suggest_float('colsample_bytree', 0.1, 1)

        model = xgb.XGBClassifier(
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            max_depth=max_depth,
            min_child_weight=min_child_weight,
            subsample=subsample,
            colsample_bytree=colsample_bytree,
            objective='multi:softmax',  # Adjust based on your problem type
            num_class=18  # Number of classes
        )

        #x_train2, x_val, y_train2, y_val = train_test_split(x_train, y_train, test_size=0.25, stratify=y_train)
        model.fit(x_train, y_train)
        y_pred = model.predict(x_val)
        score = accuracy_score(y_val, y_pred)
        return score


    study = optuna.create_study(direction='maximize')
    study.optimize(objective_XGBOOST, n_trials=100)
    start=time.time()
    best_trial = study.best_trial
    best_params = study.best_params
    best_score = best_trial.value
    best_model = xgb.XGBClassifier(**best_params, random_state=42)
    best_model.fit(x_train, y_train)
    y_pred = best_model.predict(x_test)
    test_score = accuracy_score(y_test, y_pred)
    end=time.time()
    print(end-start, " seconds")
    classification_rep = classification_report(y_test, y_pred)
    current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    filename = f"{namee}_classification_report.txt"
    with open(filename, 'w') as file:
        file.write(classification_rep)

    print(f"Classification report saved as {filename}")

    return best_params, test_score



def optimize_SVM(x_train, x_val, x_test, y_train, y_val, y_test, namee):
    def objective_SVM(trial):
        # Limit C to specific values
        C = trial.suggest_categorical('C', [0.1, 0.3, 0.6, 1])
        # Limit kernel options to faster choices
        kernel = trial.suggest_categorical('kernel', ['linear', 'rbf'])
        degree = trial.suggest_int('degree', 2, 5, log=False)
        gamma = trial.suggest_categorical('gamma', ['scale', 'auto'])

        model = SVC(C=C, kernel=kernel, degree=degree, gamma=gamma)
        model.fit(x_train, y_train)
        y_pred = model.predict(x_val)
        score = accuracy_score(y_val, y_pred)

        return score

    study = optuna.create_study(direction='maximize')
    study.optimize(objective_SVM, n_trials=100)  # Keep the same number of trials
    start = time.time()
    best_trial = study.best_trial
    best_params = study.best_params
    best_score = best_trial.value
    best_model = SVC(**best_params, random_state=42)
    best_model.fit(x_train, y_train)
    y_pred = best_model.predict(x_test)
    test_score = accuracy_score(y_test, y_pred)
    end = time.time()
    print(end - start, " seconds")
    
    classification_rep = classification_report(y_test, y_pred)
    current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    filename = f"{namee}_classification_report.txt"
    with open(filename, 'w') as file:
        file.write(classification_rep)

    print(f"Classification report saved as {filename}")

    return best_params, test_score



def NB_CLFY(x_train, x_test, y_train, y_test, namee):
    #naive bayesian
    y_train = np.array(y_train)
    
    y_test = np.array(y_test)
    model = GaussianNB()
    start=time.time()
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)
    score = accuracy_score(y_test, y_pred)
    end=time.time()
    print(end-start, " seconds")
    classification_rep = classification_report(y_test, y_pred)
    current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    filename = f"{namee}_classification_report.txt"
    with open(filename, 'w') as file:
        file.write(classification_rep)

    print(f"Classification report saved as {filename}")
    return score



def optimize_MLP(x_train, x_val, x_test, y_train, y_val, y_test, namee):
    # multi-layer perceptron
    #Just import libraries
    encoder = OneHotEncoder(sparse=False)
    y_train_en = encoder.fit_transform(np.array(y_train).reshape(-1, 1))
    y_test_en = encoder.transform(np.array(y_test).reshape(-1, 1))
    y_val_en = encoder.transform(np.array(y_val).reshape(-1, 1))
    def objective_MLP(trial):
        hidden_layer_sizes = tuple(trial.suggest_int(f'n_neurons_layer{i}', 1, 100) for i in range(trial.suggest_int('n_layers', 1, 3)))
        activation = trial.suggest_categorical('activation', ['identity', 'logistic', 'tanh', 'relu'])
        solver = trial.suggest_categorical('solver', ['lbfgs', 'sgd', 'adam'])
        alpha = trial.suggest_loguniform('alpha', 1e-5, 1e-1)

        model = MLPClassifier(hidden_layer_sizes=hidden_layer_sizes, activation=activation, 
                              solver=solver, alpha=alpha, max_iter=500)
        #x_train2, x_val, y_train2, y_val = train_test_split(x_train, y_train_en, test_size=0.25, stratify=y_train_en)

        model.fit(x_train, y_train)
        y_pred = model.predict(x_val)
        score = accuracy_score(y_val, y_pred)

        return score

    study = optuna.create_study(direction='maximize')
    study.optimize(objective_MLP, n_trials=100)
    best_trial = study.best_trial
    best_params = study.best_params
    best_score = best_trial.value

    n_neurons = []  # Initialize list to store 'n_neurons_layer{i}' values
    n_layers = best_params.pop('n_layers')

    for i in range(n_layers):
        key = f'n_neurons_layer{i}'
        n_neurons.append(best_params.pop(key))  # Remove 'n_neurons_layer{i}' key and append its value to the list

    best_params['n_neurons'] = n_neurons  # Add the list of 'n_neurons_layer{i}' values to the dictionary with key 'n_neurons'

    print(best_params)
    best_model = MLPClassifier(**best_params, random_state=42)
    start=time.time()
    best_model.fit(x_train, y_train_en)
    y_pred = best_model.predict(x_test)
    test_score = accuracy_score(y_test_en, y_pred)
    end=time.time()
    print(end-start, " seconds")
    classification_rep = classification_report(y_test, y_pred)
    current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    filename = f"{namee}_classification_report.txt"
    with open(filename, 'w') as file:
        file.write(classification_rep)

    print(f"Classification report saved as {filename}")

    return best_params, test_score











