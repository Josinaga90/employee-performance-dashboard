
import streamlit as st
import joblib
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

st.set_page_config(page_title="Employee Performance Prediction Dashboard", layout="wide")

# ================================
# LOAD MODELS
# ================================
@st.cache_resource
def load_data():
    model_RF = joblib.load("model_RF.pkl")
    model_MLR = joblib.load("model_MLR.pkl")
    model_SVMC = joblib.load("model_SVMC.pkl")
    scaler = joblib.load("scaler.pkl")
    model_features = joblib.load("model_features.pkl")

    return model_RF, model_MLR, model_SVMC, scaler, list(model_features)

#Loading result tables
@st.cache_data
def load_tables():
    model_comparison = pd.read_csv("model_comparison.csv")
    age_fairness = pd.read_csv("age_fairness.csv")
    gender_fairness = pd.read_csv("gender_fairness.csv")
    return model_comparison, age_fairness, gender_fairness

model_RF, model_MLR, model_SVMC, scaler, model_features = load_data()
model_comparison, age_fairness, gender_fairness = load_tables()

# Dashboard title
st.title("Employee Performance Prediction Dashboard")

st.write("This dashboard presents individual employee performance prediction, "
    "model comparison, and fairness analysis by gender and age.")

#Sidebar: Create interactive inputs for the HR Manager
section = st.sidebar.selectbox(
    "Select Section",
    ["Individual Prediction", "Model Comparison", "Fairness Analysis"])

#========================================================================
# 1. INDIVIDUAL PREDICTION
#========================================================================

if section == "Individual Prediction":

    st.header("Individual Employee Performance Prediction")

    selected_model = st.selectbox("Select Model",
        ["Random Forest", "Logistic Regression", "Support Vector Machine"])

    age = st.number_input("Age", min_value=18, max_value=70, value=35)
    
    tenure = st.number_input("Tenure (days in company)", min_value=0, value=1000)
    
    gender = st.selectbox("Gender", ["Male", "Female"])

    race = st.selectbox("Race", ["White", "Black", "Hispanic", "Asian", "Other"])

    department = st.selectbox("Department Type", ["Production", "Sales", "Admin Offices", "IT/IS", "Executive Office"])

    payzone = st.selectbox("Pay Zone", ["Zone A", "Zone B", "Zone C"])

    # Creatinginput data with the same columns used in training
    input_data = pd.DataFrame([[0]*len(model_features)], columns=model_features)

    # Fill numeric variables
    if "Age" in input_data.columns:
        input_data["Age"] = age
        
    if "Tenure" in input_data.columns:
        input_data["Tenure"] = tenure

    if "GenderCode" in input_data.columns:
        input_data["GenderCode"] = 0 if gender == "Male" else 1

    # Filling dummy variables
    race_column = "RaceDesc_" + race
    department_column = "DepartmentType_" + department
    payzone_column = "PayZone_" + payzone

    if race_column in input_data.columns:
        input_data[race_column] = 1

    if department_column in input_data.columns:
        input_data[department_column] = 1

    if payzone_column in input_data.columns:
        input_data[payzone_column] = 1

    input_data = input_data[model_features]

    if st.button("Predict Performance"):

        if selected_model == "Random Forest":
            prediction = model_RF.predict(input_data)[0]

        elif selected_model == "Logistic Regression":
            input_scaled = scaler.transform(input_data)
            prediction = model_MLR.predict(input_scaled)[0]

        else:
            prediction = model_SVMC.predict(scaler.transform(input_data))[0]

        performance_labels = {
            0: "PIP",
            1: "Needs Improvement",
            2: "Fully Meets",
            3: "Exceeds"}

        st.subheader("Prediction Result")
        st.success("Predicted Performance: " + performance_labels[prediction])

#========================================================================
# 2. MODEL COMPARISON
#========================================================================

elif section == "Model Comparison":

    st.header("Model Comparison")

    st.write("This section compares the performance of Random Forest, "
        "Logistic Regression, and Support Vector Machine.")
    
    model_display = model_comparison.copy()

    for col in model_display.columns[1:]:
      model_display[col] = (model_display[col]*100).round(2)

    st.subheader("Model Performance Table")
    st.dataframe(model_display)

    st.subheader("Accuracy Comparison")
    st.data_chart(model_display.set_index("Model")["Accuracy"])

    st.subheader("F1 Macro Comparison")
    st.data_chart(model_display.set_index("Model")["F1_Macro"])
    
    st.subheader("F1 Weighted Comparison")
    st.data_chart(model_display.set_index("Model")["F1_Weighted"])

#========================================================================
# 3. FAIRNESS ANALYSIS
#========================================================================

elif section == "Fairness Analysis":

    st.header("Fairness Analysis")

    st.write("This section evaluates whether the selected model performs differently "
            "across gender and age groups.")
    
    gender_display = gender_fairness.copy()
    age_display = age_fairness.copy()

    for df in [gender_display, age_display]:
      for col in df.columns[1:]:
        df[col] = df [col].round(2)

    st.subheader("Gender Fairness")
    st.dataframe(gender_display)

    st.subheader("Age Fairness")
    st.dataframe(age_display)
