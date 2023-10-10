import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image

# Add a markdown section with custom CSS
# st.markdown(
#     f"""
#     <style>
#         {open("styles.css").read()}
#     </style>
#     """,
#     unsafe_allow_html=True
# )

# The rest of your Streamlit app code goes here
model_path = 'ChestXray.h5'  # Replace with your model path
loaded_model = tf.keras.models.load_model(model_path)
class_labels = ["COVID19", "NORMAL", "PNEUMONIA", "TUBERCULOSIS"]

def add_bg_from_url():
    st.markdown(
         f"""
         <style>
         .stApp {{
             background-image: url("https://plus.unsplash.com/premium_photo-1661634247664-ac7d0f00cf0c?ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D&auto=format&fit=crop&w=1771&q=80");
             background-attachment: fixed;
             background-size: cover
         }}
         </style>
         """,
         unsafe_allow_html=True
     )
    
def homepage():

    st.title("Welcome to MedGuide")
    st.write("<span style='font-size:30px'><b>X-ray analysis for Tuberculosis, COVID-19, and Pneumonia</b></span>", unsafe_allow_html=True)

    add_bg_from_url()

# Function for the User Info Page
def user_info():
    st.title("User Information")
    name = st.text_input("Enter your name:")
    age = st.number_input("Enter your age:")
    gender = st.selectbox("Select your gender:", ["Male", "Female", "Other"])
    st.write(f"Name: {name}")
    st.write(f"Age: {age}")
    st.write(f"Gender: {gender}")

# Function for the X-ray Upload and Prognosis Page
target_size = (150, 150)  # Adjust based on your model's input shape
uploaded_image = None  # Initialize a variable to store the uploaded image
predicted_class_label = None  # Initialize a variable to store the predicted class label

def xray_upload():
    st.title("X-ray Upload and Prognosis")
    uploaded_file = st.file_uploader("Upload your X-ray image (JPEG or PNG):", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
    # Display the uploaded image
        img = image.load_img(uploaded_file, target_size=(150, 150))
        img = image.img_to_array(img)
        img = np.expand_dims(img, axis=0)

        st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)

    # Make a prediction
        predictions = loaded_model.predict(img)
        predicted_class_index = np.argmax(predictions)
        predicted_class_label = class_labels[predicted_class_index]

    # Display the predicted class label
        st.write(f"Predicted class: {predicted_class_label}")

        # # Render the progress bars
        # for i in range(len(class_labels)):
        #     class_label = class_labels[i]
        #     predicted_percentage = predictions[i] * 100
        #     st.write(f"{class_label}:")
        #     st.progress(predicted_percentage / 100.0)

# Function for the Prediction Results Page
def display_predictions_in_slider():
    st.title("Prediction Results")
    st.subheader("Predicted Class Probabilities:")
    global uploaded_image, predicted_class_label  # Access global variables
    if uploaded_image is not None:
        # Make a prediction
        predictions = loaded_model.predict(preprocess_image(uploaded_image))[0]

        for i, class_label in enumerate(class_labels):
            predicted_percentage = predictions[i] * 100
            st.write(f"{class_label}: {predicted_percentage:.2f}%")
            st.slider(f"{class_label} Likelihood", 0.0, 100.0, predicted_percentage)
        
        # Provide tips based on the predicted class label
        provide_tips(predicted_class_label)

    else:
        st.warning("Please upload an X-ray image first.")

# Function to provide tips based on the predicted class label
def provide_tips(predicted_class_label):
    st.title("Health Tips")
    if predicted_class_label == "COVID19":
        st.write("You have been predicted with COVID-19. Please follow your healthcare provider's instructions and quarantine yourself.")
        st.write("Maintain proper hygiene, wear a mask, and practice social distancing.")
    elif predicted_class_label == "NORMAL":
        st.write("Your X-ray shows no signs of any lung disease. Continue to maintain a healthy lifestyle with a balanced diet and regular exercise.")
    elif predicted_class_label == "PNEUMONIA":
        st.write("You have been predicted with Pneumonia. Please consult your doctor immediately for further evaluation and treatment.")
        st.write("Rest, stay hydrated, and follow your doctor's recommendations.")
    elif predicted_class_label == "TUBERCULOSIS":
        st.write("You have been predicted with Tuberculosis. Please consult your doctor immediately for further evaluation and treatment.")
        st.write("Follow your doctor's treatment plan and take medications as prescribed.")

# Helper function to preprocess the image
def preprocess_image(img):
    img = img.resize(target_size)  # Resize the image to match the model's input size
    img = image.img_to_array(img)
    img = np.expand_dims(img, axis=0)
    return img

# Function for the Reviews and Feedback Page
def reviews_feedback():
    st.title("Reviews and Feedback")
    feedback = st.text_area("Give us your feedback:")
    rating = st.slider("Rate your experience (1-10):", 1, 10)
    st.write(f"Feedback: {feedback}")
    st.write(f"Rating: {rating}")

# Sidebar Navigation
page = st.sidebar.selectbox("Select a page:", ["Home", "User Info", "X-ray Upload", "Reviews & Feedback"])

# Display Pages Based on Selection
if page == "Home":
    homepage()
elif page == "User Info":
    user_info()
elif page == "X-ray Upload":
    xray_upload()
# elif page == "Display Predictions":
#     display_predictions_in_slider()
elif page == "Reviews & Feedback":
    reviews_feedback()
