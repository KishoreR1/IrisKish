import streamlit as st
import numpy as np
import pickle
import requests
from sklearn.datasets import load_iris

# Load the Iris dataset (for species names)
iris = load_iris()

# Function to predict the species based on user input
def predict_species(sepal_length, sepal_width, petal_length, petal_width):
    input_features = np.array([[sepal_length, sepal_width, petal_length, petal_width]])  # Ensure only 4 features
    prediction = rf_model.predict(input_features)
    species = iris.target_names[prediction][0]  # Get species name from prediction
    return species

# 1) Correctly define the model URL with the GitHub raw URL for the model
model_url = 'https://raw.githubusercontent.com/KishoreR1/IrisKish/refs/heads/main/app.py'  # Replace with the actual URL

# 2) Download the pickled model from GitHub
response = requests.get(model_url)
with open('rf_model.pkl', 'wb') as f:
    f.write(response.content)

# 3) Load the pickled model from the file
with open('rf_model.pkl', 'rb') as f:
    rf_model = pickle.load(f)

# 4) Create the Streamlit UI
st.title("Iris Species Predictor")

# 5) Input fields for the user to enter flower measurements
sepal_length = st.number_input("Sepal Length (cm)", min_value=0.0, max_value=10.0, value=5.0)
sepal_width = st.number_input("Sepal Width (cm)", min_value=0.0, max_value=10.0, value=3.0)
petal_length = st.number_input("Petal Length (cm)", min_value=0.0, max_value=10.0, value=4.0)
petal_width = st.number_input("Petal Width (cm)", min_value=0.0, max_value=10.0, value=1.0)


# 6) Button to trigger the prediction
if st.button("Predict Species"):
    # Predict species based on user inputs
    species = predict_species(sepal_length, sepal_width, petal_length, petal_width)

 # 7) Display the predicted species
    st.write(f"The predicted species is: {species}")

