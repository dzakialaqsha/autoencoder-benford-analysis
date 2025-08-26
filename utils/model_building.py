import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import math
from scipy import stats

from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.utils import plot_model
from tensorflow.keras.losses import MeanSquaredError, Reduction
from tensorflow.data import Dataset
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.activations import leaky_relu

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, make_scorer, classification_report
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import MinMaxScaler
from functools import partial
from re import X

def build_autoencoder(input_dim, latent_dim):
  '''
  Builds a deep autoencoder model using the Keras Functional API.

    This function constructs a neural network with a symmetrical encoder-decoder
    structure. The encoder compresses the input data into a lower-dimensional
    latent space, while the decoder attempts to reconstruct the original
    input from this compressed representation.

    Args:
        input_dim (int): The dimensionality of the input data (number of features).
        latent_dim (int): The dimensionality of the latent space, which
                          represents the compressed data.

    Returns:
        keras.Model: A Keras Model object representing the complete autoencoder.
  '''
  def input_layer(x):
    return Input(shape=(input_dim,), name='Input Layer')

  def encoder_layers(input_tensor):
    x = Dense(256, name = 'encoder_1', activation=leaky_relu)(input_tensor)
    x = Dense(128, name = 'encoder_2', activation=leaky_relu)(x)
    x = Dense(64, name = 'encoder_3', activation=leaky_relu)(x)
    x = Dense(32, name ='encoder_4', activation=leaky_relu)(x)
    x = Dense(16, name ='encoder_5', activation=leaky_relu)(x)
    x = Dense(8, name ='encoder_6', activation=leaky_relu)(x)
    x = Dense(4, name ='encoder_7', activation=leaky_relu)(x)
    latent_space = Dense(latent_dim, name='Latent_Space')(x)
    return latent_space

  def decoder_layers(latent_tensor):
    x = Dense(4, name ='decoder_1', activation=leaky_relu)(latent_tensor)
    x = Dense(8, name ='decoder_2', activation=leaky_relu)(x)
    x = Dense(16, name ='decoder_3', activation=leaky_relu)(x)
    x = Dense(32, name ='decoder_4', activation=leaky_relu)(x)
    x = Dense(64, name ='decoder_5', activation=leaky_relu)(x)
    x = Dense(128, name = 'decoder_6', activation=leaky_relu)(x)
    x = Dense(256, name = 'decoder_7', activation=leaky_relu)(x)
    output_reconstruction = Dense(input_dim, activation='sigmoid', name='Reconstruction_Output')(x)
    return output_reconstruction

  def build_autoencoder_model(input_dim, latent_dim):
      input_tensor = Input(shape=(input_dim,), name='Input_Layer')
      encoded_output = encoder_layers(input_tensor)
      decoded_output = decoder_layers(encoded_output)
      autoencoder = Model(inputs=input_tensor, outputs=decoded_output, name='Functional_Autoencoder')
      return autoencoder

  model = build_autoencoder_model(input_dim, latent_dim)
  return model

def train_autoencoder(model, train_data, epochs, batch_size, learning_rate):
  '''
  '''
  if not isinstance(learning_rate, float):
    raise ValueError("Error. Provided learning_rate argument is not a float")

  if not isinstance(epochs, int):
    raise ValueError("Error. Provided epochs argument is not an integer")

  if not isinstance(batch_size, int):
    raise ValueError("Error. Provided batch_size argument is not an integer")

  if not isinstance(train_data, np.ndarray):
    raise ValueError("Error. Provided train_data argument is not a numpy array")

  if not isinstance(model, Model):
    raise ValueError("Error. Provided model argument is not a Keras Model object")

  optimizer = Adam(learning_rate=learning_rate)
  model.compile(optimizer=optimizer, loss='mean_squared_error')

  history = model.fit(train_data, train_data, epochs=15, batch_size=32)
  reconstructions = model.predict(train_data)

  mse_loss_fn = MeanSquaredError(reduction=Reduction.NONE)
  individual_mse = mse_loss_fn(train_data, reconstructions).numpy()

  fig = plt.figure(figsize=(10, 6))
  plt.plot(history.history['loss'], label='Training Loss')
  plt.xlabel('Epochs')
  plt.ylabel('Loss')
  plt.title('Training Loss')
  plt.legend()


  return history, reconstructions, individual_mse, fig

def heaviside_step_function(data, reconstruction_error_list, threshold):
  '''
  Applies a Heaviside step function to label data based on a reconstruction error threshold.

    This function classifies data points as either 0 (below or equal to the threshold) or 1
    (above the threshold), based on their reconstruction error. It adds the reconstruction
    errors and the resulting predicted labels as new columns to the input DataFrame.
  '''
  if not isinstance(data, pd.DataFrame):
    raise ValueError("Error. Provided data argument is not a Pandas DataFrame object")
  if not isinstance(reconstruction_error_list, np.ndarray):
    raise ValueError("Error. Provided reconstruction_error_list argument is not a numpy array")
  if not isinstance(threshold, float):
    raise ValueError("Error. Provided threshold argument is not a float")

  if threshold <= 0:
    raise ValueError("Error. Provided threshold argument is not a positive float")

  data = data.copy()
  data['reconstruction_error'] = reconstruction_error_list
  data['predicted_label'] = (reconstruction_error_list > threshold).astype(int)
  return data

def evaluate_prediction(data, ground_truth_column, prediction_column):
  '''
  Evaluates the performance of a binary classification model.

    This function calculates and returns several key performance metrics for a
    classification model, including accuracy, precision, recall, F1-score,
    and the confusion matrix.

    Parameters
    ----------
    data : pd.DataFrame
        The DataFrame containing the true labels and the predicted labels.
    ground_truth_column : str
        The name of the column in `data` that contains the true labels.
    prediction_column : str
        The name of the column in `data` that contains the predicted labels.

    Returns
    -------
    tuple
        A tuple containing the following metrics, in order:
        - accuracy : float
            The classification accuracy.
        - precision : float
            The precision score.
        - recall : float
            The recall score.
        - f1 : float
            The F1-score.
        - confusion_mat : np.ndarray
            The confusion matrix.
  '''
  if not isinstance(data, pd.DataFrame):
    raise ValueError("Error. Provided data argument is not a Pandas DataFrame object")
  if not isinstance(ground_truth_column, str):
    raise ValueError("Error. Provided ground_truth_column argument is not a string")
  if not isinstance(prediction_column, str):
    raise ValueError("Error. Provided prediction_column argument is not a string")
  if ground_truth_column not in data.columns:
    raise ValueError(f"Error. Provided ground_truth_column argument: {ground_truth_column} does not exist in the provided data")
  if prediction_column not in data.columns:
    raise ValueError(f"Error. Provided prediction_column argument: {prediction_column} does not exist in the provided data")

  accuracy = accuracy_score(data[ground_truth_column], data[prediction_column])
  precision = precision_score(data[ground_truth_column], data[prediction_column])
  recall = recall_score(data[ground_truth_column], data[prediction_column])
  f1 = f1_score(data[ground_truth_column], data[prediction_column])
  confusion_mat = confusion_matrix(data[ground_truth_column], data[prediction_column])

  return accuracy, precision, recall, f1, confusion_mat
