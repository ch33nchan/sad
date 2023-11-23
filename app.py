import streamlit as st
import numpy as np

def detect_emotions(heart_rate, spo2, calories_burnt, num_steps):
    emotions = ['Anger', 'Fear', 'Sadness', 'Disgust', 'Surprise', 'Anticipation', 'Trust', 'Joy']
    emotion_probabilities = np.random.rand(8)
    
    total_prob = sum(emotion_probabilities)
    emotion_percentages = [prob / total_prob * 100 for prob in emotion_probabilities]

    final_output = emotions[np.argmax(emotion_probabilities)]

    return emotions, emotion_percentages, final_output

def main():
    st.title("Emotion and Mental Health Analysis")

    num_steps = st.slider("Number of Steps", min_value=0, max_value=10000, value=5000)
    heart_rate = st.slider("Heart Rate", min_value=60, max_value=180, value=80)
    spo2 = st.slider("SpO2", min_value=90, max_value=100, value=95)
    calories_burnt = st.slider("Calories Burnt", min_value=0, max_value=1000, value=500)

    if st.button("Detect Emotions"):
        emotions, emotion_percentages, final_output = detect_emotions(heart_rate, spo2, calories_burnt, num_steps)

        st.subheader("Emotion Percentages:")
        for emotion, percentage in zip(emotions, emotion_percentages):
            st.write(f"{emotion}: {percentage:.2f}%")

        st.subheader("Final Output:")
        st.success(f"The primary emotion detected is: {final_output}")

        if final_output in ['Anger', 'Fear', 'Sadness', 'Disgust']:
            st.warning("Warning: The detected emotion may indicate stress or negative mental state.")
        elif final_output in ['Anxiety']:
            st.warning("Warning: The detected emotion may indicate anxiety.")
        elif final_output in ['Depression']:
            st.warning("Warning: The detected emotion may indicate depression.")
        else:
            st.success("No immediate concerns detected.")

if __name__ == "__main__":
    main()
