"""
Neural network and probabilistic classifiers for time series classification.

This module contains implementations of neural network based and probabilistic
classification algorithms optimized for time series classification tasks,
including hyperparameter optimization via Optuna.
"""

import time
from datetime import datetime
import numpy as np
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import OneHotEncoder
from sklearn.neural_network import MLPClassifier
import optuna
from sklearn.naive_bayes import GaussianNB


def optimize_MLP(x_train, x_val, x_test, y_train, y_val, y_test,namee, labels=None):
    # One-hot encode the target labels
    encoder = OneHotEncoder(sparse=False)
    y_train_en = encoder.fit_transform(np.array(y_train).reshape(-1, 1))
    y_test_en = encoder.transform(np.array(y_test).reshape(-1, 1))

    def objective_MLP(trial):
        hidden_layer_sizes = tuple(trial.suggest_int(f'n_neurons_layer{i}', 1, 100) for i in range(trial.suggest_int('n_layers', 1, 3)))
        activation = trial.suggest_categorical('activation', ['identity', 'logistic', 'tanh', 'relu'])
        solver = trial.suggest_categorical('solver', ['lbfgs', 'sgd', 'adam'])
        alpha = trial.suggest_loguniform('alpha', 1e-5, 1e-1)

        model = MLPClassifier(hidden_layer_sizes=hidden_layer_sizes, activation=activation, solver=solver, alpha=alpha, max_iter=500)

        model.fit(x_train, y_train_en)
        y_pred = model.predict(x_val)
        y_pred_labels = np.argmax(y_pred, axis=1)
        y_val_labels = np.argmax(encoder.transform(np.array(y_val).reshape(-1, 1)), axis=1)
        score = accuracy_score(y_val_labels, y_pred_labels)

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

    best_params['hidden_layer_sizes'] = n_neurons  # Add the list of 'n_neurons_layer{i}' values to the dictionary with key 'n_neurons'

    print(best_params)
    best_model = MLPClassifier(**best_params, random_state=42)
    best_model.fit(x_train, y_train_en)
    y_pred = best_model.predict(x_test)
    y_pred_labels = np.argmax(y_pred, axis=1)
    y_test_labels = np.argmax(y_test_en, axis=1)
    test_score = accuracy_score(y_test_labels, y_pred_labels)

    classification_rep = classification_report(y_test_labels, y_pred_labels, target_names=labels)
    current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    filename = f"{namee}_classification_report_{current_time}.txt"
    with open(filename, 'w') as file:
        file.write(classification_rep)

    print(f"Classification report saved as {filename}")

    return best_params, test_score


def optimize_NB(x_train, x_val, x_test, y_train, y_val, y_test,namee, labels=None):

    def objective_NB(trial):
        var_smoothing = trial.suggest_float('var_smoothing', 1e-12, 1e-2, log=True)

        model = GaussianNB(var_smoothing=var_smoothing)
        model.fit(x_train, y_train)
        y_pred = model.predict(x_val)
        score = accuracy_score(y_val, y_pred)

        return score

    study = optuna.create_study(direction='maximize')
    study.optimize(objective_NB, n_trials=100)
    start = time.time()
    best_trial = study.best_trial
    best_params = study.best_params
    best_score = best_trial.value

    print("Best hyperparameters: ", best_params)
    print("Best validation accuracy: ", best_score)

    best_model = GaussianNB(**best_params)
    best_model.fit(x_train, y_train)
    y_pred = best_model.predict(x_test)
    test_score = accuracy_score(y_test, y_pred)
    end = time.time()
    print(f"Test accuracy: {test_score}")
    print(f"Time taken: {end-start} seconds")

    classification_rep = classification_report(y_test, y_pred, target_names=labels)
    current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    filename = f"{namee}_classification_report_{current_time}.txt"
    with open(filename, 'w') as file:
        file.write(classification_rep)

    print(f"Classification report saved as {filename}")

    return best_params, test_score

