import streamlit as st

# Streamlit app title
st.title("Streamlit App with URL Button")

# Display some text
st.header("Welcome to my Streamlit App!")
st.write("This app demonstrates a simple Streamlit interface with a button.")

# Define the URL you want to open when the button is clicked
url = "https://large-pets-refuse.loca.lt/"

# Create a button to open the URL
if st.button("Open URL"):
    st.write(f"Opening URL: {url}")
    # Redirect to the URL when the button is clicked
    st.markdown(f'<a href="{url}" target="_blank">Click here to open the URL</a>', unsafe_allow_html=True)
