import streamlit as st
import pandas as pd
from os import path
import numpy as np
import pickle

st.title("Flower Species Predictor")

st.caption("Please provide the measurements below")

petal_length = st.number_input(
            "Petal Length",
            min_value=1.0,
            max_value=5.0,
            help="Enter a value between 1.0 and 5.0"
        )
st.caption("👉 Petal length should be between **1.0 and 5.0 cm**.")

petal_width = st.number_input(
            "Petal Width",
            min_value=0.1,
            max_value=2.5,
            help="Enter a value between 0.1 and 2.5"
        )
st.caption("👉 Petal width should be between **0.1 and 2.5 cm**.")

sepal_length = st.number_input(
            "Sepal Length",
            min_value=4.3,
            max_value=7.9,
            help="Enter a value between 4.3 and 7.9"
        )
st.caption("👉 Sepal length should be between **4.3 and 7.9 cm**.")
sepal_width = st.number_input(
            "Sepal Width",
            min_value=2.0,
            max_value=4.4,
            help="Enter a value between 2.0 and 4.4"
        )
st.caption("👉 Sepal width should be between **2.0 and 4.4 cm**.")

st.success("Great! Now you're all set to predict ")


#prepare dataframe for prediction
df_user_input=pd.DataFrame([[sepal_length, sepal_width,petal_length, petal_width]],
                        columns=['sepal_length','sepal_width','petal_length','petal_width'])
#using .pkl file,creating model named "iris_predictor"
model_path=(path.join("model","rf_model.pkl"))
with open(model_path, 'rb') as f:
    model = pickle.load(f)

    species={0:'setosa',1:'versicolor',2:'virginica'}
st.write(df_user_input)
if st.button("predict the species"):
    if((petal_length==None) or (petal_width==None) or (sepal_length==None) or (sepal_width==None)):
        st.write("please fill all values")
    else:
        # petal_length=float(petal_length)
        # petal_width=float(petal_width)
        predicted_species=model.predict(df_user_input)
        st.write("the species is ",   species[predicted_species[0]])