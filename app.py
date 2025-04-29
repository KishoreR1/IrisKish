import streamlit as st
import numpy as np
import pickle

# 1) Load model
with open('models/iris/random_forest_iris.pkl', 'rb') as f:
    model = pickle.load(f)

# 2) Streamlit title and inputs
st.title("Iris Flower Species Predictor")
sepal_length = st.number_input("Sepal Length (cm)", 0.0, 10.0, 5.0, 0.1)
sepal_width  = st.number_input("Sepal Width  (cm)", 0.0, 10.0, 3.0, 0.1)
petal_length = st.number_input("Petal Length (cm)", 0.0, 10.0, 1.5, 0.1)
petal_width  = st.number_input("Petal Width  (cm)", 0.0, 10.0, 0.3, 0.1)

# 3) Predict button and display
if st.button("Predict Iris Species"):
    # 3.1 Pack inputs and predict
    input_data = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
    pred_index = int(model.predict(input_data)[0])
    
    # 3.2 Species labels
    labels = ["Setosa", "Versicolor", "Virginica"]
    
    # 3.3 Display as colored buttons
    cols = st.columns(3)
    for idx, label in enumerate(labels):
        if idx == pred_index:
            bg, fg = "white", "black"
        else:
            bg, fg = "#e0e0e0", "#666666"
        
        btn_html = f"""
        <button style="
          background-color: {bg};
          color: {fg};
          border: 1px solid #999;
          border-radius: 4px;
          padding: 8px 16px;
          width: 100%;
          cursor: default;
        " disabled>{label}</button>
        """
        cols[idx].markdown(btn_html, unsafe_allow_html=True)


