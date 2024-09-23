import streamlit as st
import pandas as pd
import numpy as np
import math
import pickle
import sklearn


from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from arize.pandas.logger import Client, Schema
from arize.utils.types import ModelTypes, Environments, Schema, Metrics

API_KEY = '598b8202d45d6b65f70'
SPACE_KEY = 'c2f8f3b'
arize_client = Client(space_key=SPACE_KEY, api_key=API_KEY)


# Load the pre-trained models
with open(r'C:\Users\saksh\OneDrive\Documents\VIT\placement\projects\logistic_regression_model.pkl', 'rb') as model_file:
    logistic_model = pickle.load(model_file)


with open(r'C:\Users\saksh\OneDrive\Documents\VIT\placement\projects\naive_bayes_model.pkl', 'rb') as model_file:
    naive_bayes_model = pickle.load(model_file)

with open(r'C:\Users\saksh\OneDrive\Documents\VIT\placement\projects\random_forest_model.pkl', 'rb') as model_file:
    random_forest_model = pickle.load(model_file)

# Streamlit UI
st.header("Titanic Survival Prediction")

# Input fields
col1, col2, col3 = st.columns(3)
with col1:
    PassengerId = st.number_input("Passenger ID", min_value=0, step=1, value=0)
    Pclass = st.selectbox("Class of Passenger", ("1", "2", "3"))
with col2:
    Name = st.text_input("Name of Passenger")
    Sex = st.selectbox("Gender", ("Male", "Female"))
with col3:
    Age = st.number_input("Age of passenger", min_value=0, step=1)

col4, col5, col6 = st.columns(3)
with col4:
    SibSp = st.number_input("Siblings/Spouses", min_value=0, step=1)
with col5:
    Parch = st.number_input("Parents/Children", min_value=0, step=1)
with col6:
    Ticket = st.text_input("Ticket")

col7, col8 = st.columns(2)

with col7:
    Fare = st.number_input("Fare of Journey", min_value=0.0, step=1.0)
with col8:
    Embarked = st.selectbox("Picking Point", ("C", "Q", "S"))

# Prediction button
if st.button("Predict"):
    # Convert inputs to appropriate data types and handle categorical variables
    passenger_id = int(PassengerId)
    pclass = int(Pclass)
    gender = 0 if Sex == "Female" else 1
    age = math.ceil(Age)
    sibsp = int(SibSp)
    parch = int(Parch)
    fare = float(Fare)
    embarked = 2
    if Embarked == "C":
        embarked = 0
    elif Embarked == "Q":
        embarked = 1

    # Make predictions using all three models
    logistic_prediction = logistic_model.predict([[pclass, gender, age, sibsp, parch, fare, embarked]])
    naive_bayes_prediction = naive_bayes_model.predict([[pclass, gender, age, sibsp, parch, fare, embarked]])
    random_forest_prediction = random_forest_model.predict([[pclass, gender, age, sibsp, parch, fare, embarked]])

    # Display predictions
    st.markdown(f"### Predictions for Passenger '{Name}' (ID: {passenger_id}):")

    st.markdown("#### Logistic Regression Model:")
    logistic_output_label = "Survived" if logistic_prediction[0] == 1 else "Did not survive"
    st.write(f"Logistic Regression Prediction: **{logistic_output_label}**")

    st.markdown("#### Naive Bayes Model:")
    naive_bayes_output_label = "Survived" if naive_bayes_prediction[0] == 1 else "Did not survive"
    st.write(f"Naive Bayes Prediction: **{naive_bayes_output_label}**")

    st.markdown("#### Random Forest Model:")
    random_forest_output_label = "Survived" if random_forest_prediction[0] == 1 else "Did not survive"
    st.write(f"Random Forest Prediction: **{random_forest_output_label}**")

    df = pd.DataFrame({
        "PassengerId": [passenger_id],
        "Pclass": [pclass],
        "Name":[Name],
        "Sex": [gender],
        "Age": [age],
        "SibSp": [sibsp],
        "Parch": [parch],
        "Ticket":[Ticket],
        "Fare": [fare],
        "Embarked": [embarked],
        "Survived": [random_forest_prediction[0]]
    })
    
    # Define feature column names from your dataset
    feature_column_names = [
        "Pclass",        # Class of Passenger
        "Name",          # Name of Passenger
        "Sex",           # Gender
        "Age",           # Age of passenger
        "SibSp",         # Siblings/Spouses
        "Parch",         # Parents/Children
        "Ticket",        # Ticket
        "Fare",          # Fare of Journey
        "Embarked"       # Picking Point
    ]

    # Define schema
    schema = Schema(
        prediction_id_column_name="PassengerId",        # Unique identifier for each prediction
        prediction_label_column_name="Survived",
        actual_label_column_name="Survived",             # Actual label column name (assuming binary classification)
        feature_column_names=feature_column_names,
    
    )


    # Log data to Arize AI
    response = arize_client.log(
        dataframe=df,
        schema=schema,
        model_id="titanic_survival_prediction",
        model_version="1.0.0",
        model_type=ModelTypes.BINARY_CLASSIFICATION,
        metrics_validation=[Metrics.CLASSIFICATION],
        validate=True,
        environment=Environments.PRODUCTION
    )

    if response.status_code == 200:
        st.success("Data uploaded to Arize AI successfully!")
    else:
        st.error(f"Failed to upload data to Arize AI: {response.text}")
