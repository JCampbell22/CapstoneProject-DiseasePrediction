import streamlit as st
from PIL import Image
st.title("Medical Chatbot")

uploaded_image = st.file_uploader("Upload an image for skin cancer detection")
if uploaded_image is not None:
    img = Image.open(uploaded_image)
    st.image(img, caption = "Upload Image", use_column_width = True)
    #Prediction = predict_skin_cancer(uploaded_image), this is in progress 
    #st.write(f"Prediciton: {'Cancerous' if Prediction else 'Non-Cancerous'}")