import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import os
import pickle

# Load the model using pickle
@st.cache_resource
def load_model():
    model_path = r'C:\Users\asus\Downloads\teeth classfication\teeth_classification_model.pkl'
    if not os.path.exists(model_path):
        st.error(f"Model not found at {model_path}. Please place it in the correct folder.")
        st.stop()
    
    try:
        with open(model_path, 'rb') as f:
            data = pickle.load(f)
        model = data['model']
        class_names = data['class_names']
        st.success("Model loaded successfully!")
        return model, class_names
    except Exception as e:
        st.error(f"Error loading model: {str(e)}. Ensure TensorFlow 2.19.0 is used and resave the model if needed.")
        st.stop()

model, class_names = load_model()

# Preprocess the uploaded image
def preprocess_image(image):
    image = image.resize((150, 150))
    image = np.array(image) / 255.0
    if image.shape[-1] == 4:  # Handle RGBA
        image = image[..., :3]  # Convert to RGB
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    return image

# Process multiple test images
def process_test_images(uploaded_files):
    predictions = []
    images = []
    for uploaded_file in uploaded_files:
        image = Image.open(uploaded_file)
        images.append(image)
        processed_image = preprocess_image(image)
        prediction = model.predict(processed_image)
        predictions.append(prediction)
    return images, predictions

# Main Streamlit app
st.title("Dental Image Classifier")

# Single image uploader
st.header("Single Image Prediction")
uploaded_file = st.file_uploader("Upload a dental image", type=["jpg", "jpeg", "png"], key="single_upload")

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)
    processed_image = preprocess_image(image)
    prediction = model.predict(processed_image)
    predicted_class = np.argmax(prediction, axis=1)[0]
    confidence = np.max(prediction) * 100
    st.write(f"Predicted Class: {class_names[predicted_class]}")
    st.write(f"Confidence: {confidence:.2f}%")
    st.write("Class Probabilities:")
    for i, prob in enumerate(prediction[0]):
        st.write(f"{class_names[i]}: {prob * 100:.2f}%")

# Multiple test images uploader
st.header("Test Multiple Images")
uploaded_files = st.file_uploader("Upload test images", type=["jpg", "jpeg", "png"], accept_multiple_files=True, key="multi_upload")

if uploaded_files is not None and len(uploaded_files) > 0:
    images, predictions = process_test_images(uploaded_files)
    st.write(f"Processing {len(uploaded_files)} test images:")
    for i, (image, prediction) in enumerate(zip(images, predictions)):
        st.image(image, caption=f"Test Image {i+1}", use_column_width=True)
        predicted_class = np.argmax(prediction, axis=1)[0]
        confidence = np.max(prediction) * 100
        st.write(f"Image {i+1} - Predicted Class: {class_names[predicted_class]}")
        st.write(f"Confidence: {confidence:.2f}%")
        st.write("Class Probabilities:")
        for j, prob in enumerate(prediction[0]):
            st.write(f"{class_names[j]}: {prob * 100:.2f}%")
        st.write("---")

# Add a note
st.write("Note: Ensure the model 'teeth_classification_model.pkl' is in C:\\Users\\asus\\Downloads\\teeth classfication\\.")