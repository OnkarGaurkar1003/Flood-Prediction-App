import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, Binarizer
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Dropout, LSTM, SimpleRNN, Concatenate
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import pickle  # For saving and loading scaler


scaler_filename = r'C:\Users\ASUS\Downloads\scaler.pkl'
model_filename = r'C:\Users\ASUS\Downloads\hybrid_model.h5'

with open(scaler_filename, 'rb') as f:
    loaded_scaler = pickle.load(f)

# To load the saved model later
loaded_model = tf.keras.models.load_model(model_filename)

# -------- Predicting with Loaded Model and Scaler --------

def predict(input_data):
    # Scale the input data using the loaded scaler
    input_data_scaled = loaded_scaler.transform(np.array(input_data).reshape(1, -1))
    
    # Reshape the scaled data for the LSTM model
    input_data_scaled = input_data_scaled.reshape(input_data_scaled.shape[0], input_data_scaled.shape[1], 1)
    
    # Make predictions using the loaded model
    prediction = loaded_model.predict(input_data_scaled)
    
    return prediction

# Example usage for making predictions with the loaded scaler and model
example_input = [4, 6, 3, 5, 6, 7, 8, 7, 8, 4, 8, 5, 7, 5, 6, 3, 6, 4, 4, 5]
prediction = predict(example_input)
print(f"Prediction for input: {prediction}")

if(prediction>0.5):
    print(f"High Chance of Flood: {prediction}")
else:
    print(f"Low Chance of Flood: {prediction}")

