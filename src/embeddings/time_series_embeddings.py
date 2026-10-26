"""
Time Series Embedding Methods for Classification Tasks.

This module provides implementations of various embedding techniques for time series data,
organized according to the taxonomy presented in our paper.
"""

import pandas as pd
import numpy as np
import os
import random
import matplotlib.pyplot as plt
import optuna
import time

from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.manifold import LocallyLinearEmbedding
from pywt import wavedec
from scipy.fft import fft
from scipy.spatial.distance import euclidean
from fastdtw import fastdtw
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv1D, GlobalAveragePooling1D
from sklearn.metrics import mean_absolute_error
import tensorflow as tf
from tensorflow.keras.layers import *
from tensorflow.keras.models import *
from tensorflow.keras.callbacks import *
from tensorflow.keras.optimizers import *
from tensorflow.keras import backend as K
import keras_tuner as kt
import numpy as np
from tensorflow.keras.layers import Input, Layer
from tensorflow.keras.models import Model
from gtda.time_series import TakensEmbedding
import gtda.graphs as gr
import umap.umap_ as umap
import networkx as nx
from gtda.homology import VietorisRipsPersistence
from gtda.diagrams import Amplitude
from sklearn.decomposition import PCA


import ts2vg

from keras.layers import Input, Dense, Conv1D, MaxPooling1D, Flatten, Dropout, concatenate
from tensorflow.keras.models import Model


from scipy.stats import kurtosis, skew
import networkx.algorithms.community as community

import gudhi
import numpy as np
from gudhi.wasserstein import wasserstein_distance
from gudhi.bottleneck import bottleneck_distance
from gudhi.representations import Landscape


#creating the overlapping steps

def generate_ol_steps(data, window_size, overlap):

    steps = []
    start = 0
    end = window_size
    while end <= len(data):
        steps.append((start, end))
        start += window_size - overlap
        end = start + window_size
    return steps

def create_time_steps_dataframe(x,y, window_size, overlap, fl):

    # Generate overlapping time steps
    time_steps = generate_ol_steps(x, window_size, overlap)

    # Create a new dataframe with rows for each time step
    new_x= []
    new_y=[]
    for start, end in time_steps:
        time_step_data = x.iloc[start:end]
        new_x.append({
            #'time_step_start': start,
            #'time_step_end': end,
            'data_matrix': time_step_data.values
        })

        time_step_data_yy = y.iloc[start:end]
        new_y.append({
             #time_step_data_yy = mode(time_step_data_y.values.flatten())[0][0]
            #'time_step_start': start,
            #'time_step_end': end,
            'data_matrix': time_step_data_yy.values
        })

    # Create the new dataframe
    df_x_n = pd.DataFrame(new_x)
    df_y_n = pd.DataFrame(new_y)


    def flatten_list_of_lists(lst):
      return [item for sublist in lst for item in sublist]
    def compute_mode(lst):
      mode_value = max(set(lst), key=lst.count)
      return mode_value

    if fl==1:
      df_x_n['data_matrix'] = df_x_n['data_matrix'].apply(flatten_list_of_lists)
      temp_df = df_x_n.drop('data_matrix', axis=1)
      data_df = pd.DataFrame(df_x_n['data_matrix'].to_list())
      df_x_nn = pd.concat([temp_df, data_df], axis=1)
    else:
      df_x_nn=df_x_n

    # Apply the function to each row of the DataFrame
    df_y_n['data_matrix'] = df_y_n['data_matrix'].apply(flatten_list_of_lists)
    df_y_n['data_matrix'] = df_y_n['data_matrix'].apply(compute_mode)

    return df_x_nn,df_y_n



#Now  making the encoding methods and defining them

def std_scaling(train_df,valid_df,test_df):
    #from sklearn.preprocessing #import StandardScaler
    scaler = StandardScaler()
    #first fit and transforming the train dataset
    scaled_data = scaler.fit_transform(train_df)
    train_sc = pd.DataFrame(scaled_data, columns=train_df.columns)
    scaled_valid_data = scaler.transform(valid_df)
    valid_sc = pd.DataFrame(scaled_valid_data, columns=valid_df.columns)
    #now transforming the test dataset
    scaled_test_data = scaler.transform(test_df)
    test_sc = pd.DataFrame(scaled_test_data, columns=test_df.columns)
    #getting the scaled datasets
    return train_sc, valid_sc, test_sc

def minmax_scaling(train_df,valid_df,test_df): 
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(train_df)
    train_sc = pd.DataFrame(scaled_data, columns=train_df.columns)
    scaled_valid_data = scaler.transform(valid_df)
    valid_sc = pd.DataFrame(scaled_valid_data, columns=valid_df.columns)
    #now transforming the test dataset
    scaled_test_data = scaler.transform(test_df)
    test_sc = pd.DataFrame(scaled_test_data, columns=test_df.columns)
    #getting the scaled datasets
    return train_sc, valid_sc, test_sc

def pca_embedding(train_sc,valid_sc, test_sc,n_comps):
    #first the standard scaling should be done for this method
    #from sklearn.decomposition #import PCA
    pca = PCA(n_components=n_comps)
    train_pca=pca.fit_transform(train_sc)
    valid_pca=pca.transform(valid_sc)
    test_pca=pca.transform(test_sc)
    #getting the embedded datasets
    return train_pca, valid_pca, test_pca

def lle_embedding(train_sc,valid_sc,test_sc,lle_parameters):
    #now for the Locally Linear Embedding method
    #from sklearn.manifold #import LocallyLinearEmbedding
    embedding = LocallyLinearEmbedding(
        n_neighbors=lle_parameters['n_neighbors'],
        n_components=lle_parameters['n_components'],
        random_state=lle_parameters['random_state'])
    train_lle=embedding.fit_transform(train_sc)
    valid_lle=embedding.transform(valid_sc)
    test_lle=embedding.transform(test_sc)

    return train_lle,valid_lle,test_lle

def umap_embedding(train_df,val_df,test_df):
    umap_model = umap.UMAP()
    train_umap = umap_model.fit_transform(train_df)
    val_umap = umap_model.transform(val_df)
    # Transform test data using the trained UMAP model
    test_umap = umap_model.transform(test_df)

    return train_umap, val_umap, test_umap    

def wavelet_embedding(train_df,valid_df,test_df):
    #now for the Wavelet Transform Embedding method
    #from pywt #import wavedec
    train_dwt=wavedec(train_df, 'db1', level=1)[0]
    valid_dwt=wavedec(valid_df, 'db1', level=1)[0]
    test_dwt=wavedec(test_df, 'db1', level=1)[0]
    return train_dwt,valid_dwt, test_dwt


def fft_embedding(train_df,valid_df, test_df):
    #now for the Fast Fourier transform Embedding method
    #from scipy.fft #import fft
    #the scipy fft function is more comprehensive
    train_fft = np.apply_along_axis(lambda x: np.abs(fft(x)), axis=1, arr=train_df)
    valid_fft = np.apply_along_axis(lambda x: np.abs(fft(x)), axis=1, arr=valid_df)
    test_fft = np.apply_along_axis(lambda x: np.abs(fft(x)), axis=1, arr=test_df)
    return train_fft, valid_fft, test_fft


def build_autoencoder(input_dim):
    #from tensorflow.keras.models #import Sequential
    #from tensorflow.keras.layers #import Dense
    model = Sequential()
    model.add(Dense(32, activation='relu', input_dim=input_dim))
    model.add(Dense(16, activation='relu'))
    model.add(Dense(8, activation='relu'))
    model.add(Dense(4, activation='relu'))
    model.add(Dense(8, activation='relu'))
    model.add(Dense(16, activation='relu'))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(input_dim, activation='linear'))
    model.compile(optimizer='adam', loss='mse')
    return model

def train_autoencoder(X_train):
    #from tensorflow.keras.models #import Sequential
    input_dim = X_train.shape[1]
    model = build_autoencoder(input_dim)
    model.fit(X_train, X_train, epochs=50, batch_size=32, shuffle=True, validation_split=0.2, verbose=0)
    encoder = Sequential(model.layers[:4])  # Extract the encoder part
    return encoder


def AE_embedding(train_sc,valid_sc,test_sc):
    #now for the autoencoder method

    embedding= train_autoencoder(train_sc)
    train_ae=embedding.predict(train_sc)
    test_ae=embedding.predict(test_sc)
    valid_ae=embedding.predict(valid_sc)

    return train_ae,valid_ae,test_ae



# Function to compute statistical properties of weights
def compute_weight_statistics(weights):
    statistics = {}
    weights_array = np.array(weights)
    weights_array_nonzero = weights_array[weights_array > 0]
    weights_sum = np.sum(weights_array_nonzero)
    probabilities = weights_array_nonzero / weights_sum
    entropy = -np.sum(probabilities * np.log2(probabilities))
    statistics['entropy'] = entropy

    #statistics['entropy'] = -np.sum((weights_array / np.sum(weights_array)) * np.log2(weights_array / np.sum(weights_array)))
    statistics['variance'] = np.var(weights_array)
    statistics['standard_deviation'] = np.std(weights_array)
    statistics['mean'] = np.mean(weights_array)
    statistics['median'] = np.median(weights_array)
    statistics['5th_percentile'] = np.percentile(weights_array, 5)
    statistics['25th_percentile'] = np.percentile(weights_array, 25)
    statistics['75th_percentile'] = np.percentile(weights_array, 75)
    statistics['95th_percentile'] = np.percentile(weights_array, 95)
    statistics['RMS'] = np.sqrt(np.mean(weights_array**2))
    statistics['kurtosis'] = kurtosis(weights_array)
    statistics['skewness'] = skew(weights_array)

    # Calculate zero crossings
    zero_crossings = np.where(np.diff(np.sign(weights_array)))[0]
    statistics['zero_crossings'] = len(zero_crossings)

    # Calculate mean crossings
    mean_value = np.mean(weights_array)
    mean_crossings = np.where(np.diff(np.sign(weights_array - mean_value)))[0]
    statistics['mean_crossings'] = len(mean_crossings)

    return statistics



def compute_graph_metrics(G):
    metrics = {}
    metrics['average_degree'] = np.mean([d for n, d in G.degree()])
    metrics['average_shortest_path_length'] = nx.average_shortest_path_length(G) if nx.is_connected(G) else float('inf')
    metrics['diameter'] = nx.diameter(G) if nx.is_connected(G) else float('inf')
    metrics['global_efficiency'] = nx.global_efficiency(G)
    metrics['average_clustering_coefficient'] = nx.average_clustering(G)
    metrics['degree_assortativity_coefficient'] = nx.degree_assortativity_coefficient(G)
    metrics['density'] = nx.density(G)
    metrics['transitivity'] = nx.transitivity(G)
    partition = community.greedy_modularity_communities(G)
    modularity = community.modularity(G, partition)
    metrics['modularity'] = modularity

    weights = [data['weight'] for u, v, data in G.edges(data=True)]
    weight_statistics = compute_weight_statistics(weights)

    # Combine metrics and weight statistics
    metrics.update(weight_statistics)

    return metrics


def create_vg(data):
  all_metrics = []
  for i in range(len(data)):
    vg = ts2vg.NaturalVG(weighted="abs_angle")
    vg.build(data[i])
    g=vg.as_networkx()
    metric=compute_graph_metrics(g)
    all_metrics.append(metric)
  df=pd.DataFrame(all_metrics)
  return df

def graph_embedding(train_sc,valid_sc,test_sc):
    train_gr=create_vg(train_sc)
    val_gr=create_vg(valid_sc)
    test_gr=create_vg(test_sc)
    return train_gr, val_gr, test_gr


# Example time series data (replace with your actual data)
#time_series = np.array(x_train.iloc[0])

def feature_vector_tda(time_series):

  # Step 1: Compute the persistence diagram using sublevel set filtration
  cubical_complex = gudhi.CubicalComplex(dimensions=[len(time_series)], top_dimensional_cells=time_series)
  cubical_complex.compute_persistence()
  persistence_diagram = cubical_complex.persistence_intervals_in_dimension(0)

  # Step 2: Filter out points with infinite lifetimes
  filtered_persistence_diagram = np.array([point for point in persistence_diagram if point[1] != float('inf')])

  # If the filtered diagram is empty, handle this case appropriately
  if len(filtered_persistence_diagram) == 0:
      print("No valid points in the persistence diagram after filtering out infinite lifetimes.")
  else:
      # Step 3: Compute the null diagram (diagonal)
      max_death = max(max(death for birth, death in filtered_persistence_diagram), max(time_series))
      null_diagram = np.array([[x, x] for x in np.linspace(0, max_death * 1.1, 100)])

      # Ensure both diagrams have the same number of points
      max_points = max(len(filtered_persistence_diagram), len(null_diagram))
      filtered_persistence_diagram = np.pad(filtered_persistence_diagram, ((0, max_points - len(filtered_persistence_diagram)), (0, 0)), mode='constant', constant_values=(max_death * 1.1,))
      null_diagram = np.pad(null_diagram, ((0, max_points - len(null_diagram)), (0, 0)), mode='constant', constant_values=(max_death * 1.1,))

      # Step 4: Compute the 1-Wasserstein distance and bottleneck distance
      one_wasserstein_distance = wasserstein_distance(filtered_persistence_diagram, null_diagram, order=1, keep_essential_parts=False)
      bottleneck_distance_value = bottleneck_distance(filtered_persistence_diagram, null_diagram)

      # Step 5: Compute additional features
      # Persistence entropy
      lifetimes = filtered_persistence_diagram[:, 1] - filtered_persistence_diagram[:, 0]
      total_lifetime = np.sum(lifetimes)
      probabilities = lifetimes / total_lifetime
      persistence_entropy = -np.sum(probabilities * np.log(probabilities + 1e-10))

      # Betti curve (simplified)
      betti_curve = np.sum((filtered_persistence_diagram[:, 0] <= np.arange(len(time_series))[:, None]) &
                          (np.arange(len(time_series))[:, None] < filtered_persistence_diagram[:, 1]), axis=1)
      betti_l1_norm = np.sum(betti_curve)
      betti_l2_norm = np.sqrt(np.sum(betti_curve**2))

      # Compute landscape norms
      landscape = Landscape(resolution=1000)
      landscape.fit([filtered_persistence_diagram])
      landscape_vals = landscape.transform([filtered_persistence_diagram])[0]
      landscape_l1_norm = np.sum(np.abs(landscape_vals))
      landscape_l2_norm = np.sqrt(np.sum(landscape_vals**2))

      # Form the feature vector
      feature_vector = [
          one_wasserstein_distance,
          bottleneck_distance_value,
          persistence_entropy,
          betti_l1_norm,
          betti_l2_norm,
          landscape_l1_norm,
          landscape_l2_norm
      ]

      return feature_vector


def features_tda(df_list):
    features = []
    for df in df_list:
        features.append(feature_vector_tda(df))
    return features


def create_vg_tda(time_series, graph_type):
  if graph_type=='natural':
    vg = ts2vg.NaturalVG(weighted="abs_angle")
    vg.build(time_series)
  elif graph_type=='horizontal':
    vg = ts2vg.HorizontalVG(weighted="abs_angle")
    vg.build(time_series)
  g=vg.as_networkx()
  vg_degrees = np.array([degree for node, degree in sorted(g.degree())])
  G1_n= vg_degrees.tolist()

  return G1_n


def features_vg_tda(data,graph_type):
  metric_a2=[]
  for i in range(len(data)):
    g=create_vg_tda(data[i],graph_type)
    metric=feature_vector_tda(g)
    metric_a2.append(metric)
  df=pd.DataFrame(metric_a2)
  return df

# Function to compute statistical properties of weights
def compute_weight_statistics_tda(weights):
    statistics = {}
    weights_array = np.array(weights)
    weights_array_nonzero = weights_array[weights_array > 0]
    weights_sum = np.sum(weights_array_nonzero)
    probabilities = weights_array_nonzero / weights_sum
    entropy = -np.sum(probabilities * np.log2(probabilities))
    statistics['entropy'] = entropy

    #statistics['entropy'] = -np.sum((weights_array / np.sum(weights_array)) * np.log2(weights_array / np.sum(weights_array)))
    statistics['variance'] = np.var(weights_array)
    statistics['standard_deviation'] = np.std(weights_array)
    statistics['mean'] = np.mean(weights_array)
    statistics['median'] = np.median(weights_array)
    statistics['5th_percentile'] = np.percentile(weights_array, 5)
    statistics['25th_percentile'] = np.percentile(weights_array, 25)
    statistics['75th_percentile'] = np.percentile(weights_array, 75)
    statistics['95th_percentile'] = np.percentile(weights_array, 95)
    statistics['RMS'] = np.sqrt(np.mean(weights_array**2))
    statistics['kurtosis'] = kurtosis(weights_array)
    statistics['skewness'] = skew(weights_array)

    # Calculate zero crossings
    zero_crossings = np.where(np.diff(np.sign(weights_array)))[0]
    statistics['zero_crossings'] = len(zero_crossings)

    # Calculate mean crossings
    mean_value = np.mean(weights_array)
    mean_crossings = np.where(np.diff(np.sign(weights_array - mean_value)))[0]
    statistics['mean_crossings'] = len(mean_crossings)

    return statistics
def compute_graph_metrics_tda(G):
    metrics = {}
    weights = [data['weight'] for u, v, data in G.edges(data=True)]
    weight_statistics = compute_weight_statistics_tda(weights)

    # Combine metrics and weight statistics
    metrics.update(weight_statistics)

    return metrics


def create_vg2_tda(data, graph_type):
  all_metrics = []
  for i in range(len(data)):
    if graph_type=='natural':
      vg = ts2vg.NaturalVG(weighted="abs_angle")
    elif graph_type=='horizontal':
      vg = ts2vg.HorizontalVG(weighted="abs_angle")
    vg.build(data[i])
    g=vg.as_networkx()
    metric=compute_graph_metrics_tda(g)
    all_metrics.append(metric)
  df=pd.DataFrame(all_metrics)
  return df

def merging_tda(df1,df2,df3,df4,df5):
  result_df = pd.concat([df1, df2, df3,df4,df5], axis=1)

  # Optional: Reset column names to avoid duplicates
  result_df.columns = range(result_df.shape[1])
  return result_df

def std_scaling_tda(train_df,valid_df,test_df): #CHANGE_NAME
    #from sklearn.preprocessing #import StandardScaler
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(train_df)
    train_sc = pd.DataFrame(scaled_data, columns=train_df.columns)
    scaled_valid_data = scaler.transform(valid_df)
    valid_sc = pd.DataFrame(scaled_valid_data, columns=valid_df.columns)
    #now transforming the test dataset
    scaled_test_data = scaler.transform(test_df)
    test_sc = pd.DataFrame(scaled_test_data, columns=test_df.columns)
    #getting the scaled datasets
    return train_sc, valid_sc, test_sc


#normalizing the dataset:
def minmax_scaling_tda(train_df,valid_df,test_df): #CHANGE_NAME
    #from sklearn.preprocessing #import StandardScaler
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(train_df)
    train_sc = pd.DataFrame(scaled_data, columns=train_df.columns)
    scaled_valid_data = scaler.transform(valid_df)
    valid_sc = pd.DataFrame(scaled_valid_data, columns=valid_df.columns)
    #now transforming the test dataset
    scaled_test_data = scaler.transform(test_df)
    test_sc = pd.DataFrame(scaled_test_data, columns=test_df.columns)
    #getting the scaled datasets
    return train_sc, valid_sc, test_sc


def tda_embedding(x_train,x_val,x_test):
    x_train_a1=features_tda(x_train)
    x_train_a1=pd.DataFrame(x_train_a1)
    x_train_a2=features_vg_tda(x_train,'natural')
    x_train_a22= np.array(create_vg2_tda(x_train, 'natural'))
    train_reshaped_a22 = x_train_a22.reshape(x_train_a22.shape[0], -1)
    x_train_a22 = pd.DataFrame(train_reshaped_a22)
    x_train_a3=features_vg_tda(x_train,'horizontal')
    x_train_a32= np.array(create_vg2_tda(x_train, 'horizontal'))
    train_reshaped_a32 = x_train_a32.reshape(x_train_a32.shape[0], -1)
    x_train_a32 = pd.DataFrame(train_reshaped_a32)
    
    x_train_tda=merging_tda(x_train_a1,x_train_a2,x_train_a22,x_train_a3, x_train_a32)
    
    x_val_a1=features_tda(x_val)
    x_val_a2=features_vg_tda(x_val,'natural')
    x_val_a3=features_vg_tda(x_val,'horizontal')
    x_val_a22= np.array(create_vg2_tda(x_val, 'natural'))
    val_reshaped_a22 = x_val_a22.reshape(x_val_a22.shape[0], -1)
    x_val_a22 = pd.DataFrame(val_reshaped_a22)
    x_val_a32= np.array(create_vg2_tda(x_val, 'horizontal'))
    val_reshaped_a32 = x_val_a32.reshape(x_val_a32.shape[0], -1)
    x_val_a32 = pd.DataFrame(val_reshaped_a22)
    x_val_a1=pd.DataFrame(x_val_a1)
    
    x_val_tda=merging_tda(x_val_a1,x_val_a2,x_val_a22,x_val_a3, x_val_a32)
    
    x_test_a1=features_tda(x_test)
    x_test_a2=features_vg_tda(x_test,'natural')
    x_test_a3=features_vg_tda(x_test,'horizontal')
    x_test_a22= np.array(create_vg2_tda(x_test, 'natural'))
    test_reshaped_a22 = x_test_a22.reshape(x_test_a22.shape[0], -1)
    x_test_a22 = pd.DataFrame(test_reshaped_a22)
    x_test_a32= np.array(create_vg2_tda(x_test, 'horizontal'))
    test_reshaped_a32 = x_test_a32.reshape(x_test_a32.shape[0], -1)
    x_test_a32 = pd.DataFrame(test_reshaped_a32)
    x_test_a1=pd.DataFrame(x_test_a1)
    x_test_tda=merging_tda(x_test_a1,x_test_a2,x_test_a22,x_test_a3, x_test_a32)
    return x_train_tda, x_val_tda, x_test_tda 