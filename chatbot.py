import librosa
import numpy as np
import joblib
import pandas as pd
import os
import pickle  # Importing pickle as an alternative for loading models
from sklearn.preprocessing import StandardScaler

# Try to load the model using joblib first
try:
    model = joblib.load('saved_model.pth')
    scaler = joblib.load('scaler.pkl')
except Exception as e:
    print(f"Failed to load model with joblib: {e}. Trying to load with pickle.")
    # If joblib fails, try loading with pickle
    with open('saved_model.pkl', 'rb') as f:
        model = pickle.load(f)
    with open('scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)

# Define feature columns
feature_columns = ['chroma_stft', 'rmse', 'spectral_centroid', 'spectral_bandwidth',
                   'rolloff', 'zero_crossing_rate'] + [f'mfcc{i}' for i in range(1, 21)]

# Function to extract audio features
def extract_features(file_path):
    y, sr = librosa.load(file_path, sr=None)

    chroma_stft = np.mean(librosa.feature.chroma_stft(y=y, sr=sr))
    rmse = np.mean(librosa.feature.rms(y=y))
    spectral_centroid = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr))
    spectral_bandwidth = np.mean(librosa.feature.spectral_bandwidth(y=y, sr=sr))
    rolloff = np.mean(librosa.feature.spectral_rolloff(y=y, sr=sr))
    zero_crossing_rate = np.mean(librosa.feature.zero_crossing_rate(y))
    mfcc = np.mean(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20), axis=1)

    features = np.array([chroma_stft, rmse, spectral_centroid, spectral_bandwidth, rolloff, zero_crossing_rate] + mfcc.tolist())
    return features

# Function to predict mood
def predict_mood(file_path):
    features = extract_features(file_path)
    features_df = pd.DataFrame([features], columns=feature_columns)
    features_scaled = scaler.transform(features_df)
    mood = model.predict(features_scaled)
    return mood[0]

# Simple chatbot function
def chatbot():
    print("Welcome to the Mood Detection Chatbot!")
    print("Please upload an audio file in WAV or MP3 format (type 'exit' to quit).")
    
    while True:
        user_input = input("Upload audio file: ")
        if user_input.lower() == 'exit':
            break
        
        file_path = user_input.strip('"')
        
        if not os.path.isfile(file_path):
            print("Error: The file does not exist. Please try again.")
            continue

        try:
            mood = predict_mood(file_path)
            print(f"The predicted mood is: {mood}")
        except Exception as e:
            print(f"Error: {e}")

if __name__ == "__main__":
    chatbot()
