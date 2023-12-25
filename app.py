import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

# Sample data for demonstration
data = {
    'Number of Steps': [5000, 10000, 15000, 20000],
    'spO2 Levels': [95, 97, 98, 99],
    'Heart Rate': [70, 80, 90, 100],
    'Calories Burnt': [200, 400, 600, 800],
    'Emotion_Anger': [10, 20, 30, 40],
    'Emotion_Anticipation': [5, 15, 25, 35],
    'Emotion_Joy': [30, 40, 50, 60],
    'Emotion_Trust': [50, 60, 70, 80],
    'Emotion_Fear': [15, 25, 35, 45],
    'Emotion_Surprise': [20, 30, 40, 50],
    'Emotion_Sadness': [25, 35, 45, 55],
    'Emotion_Disgust': [10, 20, 30, 40]
}

df = pd.DataFrame(data)

# Preprocess the data
X = df[['Number of Steps', 'spO2 Levels', 'Heart Rate', 'Calories Burnt']]
y = df[['Emotion_Sadness', 'Emotion_Anticipation', 'Emotion_Joy', 'Emotion_Trust',
        'Emotion_Fear', 'Emotion_Surprise', 'Emotion_Sadness', 'Emotion_Disgust']]

scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Train a simple RandomForestClassifier for demonstration
clf = RandomForestClassifier()
clf.fit(X_train, y_train)

# Streamlit UI
st.title('Emotion Prediction App')

# User input
steps = st.slider('Number of Steps taken:', min_value=0, max_value=30000, value=15000)
spo2 = st.slider('spO2 levels:', min_value=90, max_value=100, value=95)
heart_rate = st.slider('Heart Rate:', min_value=60, max_value=120, value=80)
calories_burnt = st.slider('Calories Burnt:', min_value=0, max_value=1000, value=500)

# Predict emotion percentages
input_data = scaler.transform([[steps, spo2, heart_rate, calories_burnt]])
emotion_percentages = clf.predict_proba(input_data)[0]

# Display emotion percentages
st.write('Emotion Percentages:')
for i, emotion in enumerate(['Anger', 'Anticipation', 'Joy', 'Trust', 'Fear', 'Surprise', 'Sadness', 'Disgust']):
    st.write(f'{emotion}: {emotion_percentages[i]:.2f}%')

# Predict mental health condition
st.title('Mental Health Prediction')
mental_health_prediction = clf.predict(input_data)[0]

if mental_health_prediction == 1:
    st.warning('Possible Sadness')
elif mental_health_prediction == 2:
    st.error('Possible Anxiety')
elif mental_health_prediction == 3:
    st.error('Possible Depression')
else:
    st.success('No specific mental health condition predicted')
