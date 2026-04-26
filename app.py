
import streamlit as st #Creating website
import joblib #Downloading documents as ".pkl"
import pandas as pd #Creating graphs / charts
import matplotlib.pyplot as plt #Design and drawing tools
import seaborn as sns #Design and drawing tools
import gdown
import os

#Webside Design: Set the app to wide mode for a more professional appearance
st.set_page_config(page_title="Employee Performance Prediction Dashboard", layout="wide")

@st.cache_resource
def load_models():

    # DESCARGAR SOLO EL MODELO GRANDE (RF)
    if not os.path.exists("model_RF.pkl"):
        gdown.download("https://drive.google.com/uc?id=1ts197dJz7gO12A_oDWFuX4D45WW_skYj",
            "model_RF.pkl", quiet=False)

#Loading Models' Files / Scaler / Features
@st.cache_resource
def load_data():
  model_RF = joblib.load("model_RF.pkl")
  model_MLR = joblib.load("model_MLR.pkl")
  model_SVMC = joblib.load("model_SVMC.pkl")
  scaler = joblib.load("scaler.pkl")
  model_features = joblib.load("model_features.pkl")
  return model_RF, model_MLR, model_SVMC, scaler, model_features

model_RF, model_MLR, model_SVMC, scaler, model_features = load_data()

#Loading result tables
@st.cache_data
def load_tables():
  model_comparison = pd.read_csv("model_comparison.csv")
  age_fairness = pd.read_csv("age_fairness.csv")
  gender_fairness = pd.read_csv("gender_fairness.csv")
  return model_comparison, age_fairness, gender_fairness

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

    age = st.number_input("Age", min_value=20, max_value=70, value=35)

    current_rating = st.number_input("Current Employee Rating", min_value=1, max_value=5, value=3)

    gender = st.selectbox("Gender", ["Male", "Female"])

    race = st.selectbox("Race", ["White", "Black", "Hispanic", "Asian", "Other"])

    department = st.selectbox("Department Type", ["Production", "Sales", "Admin Offices", "IT/IS", "Executive Office"])

    payzone = st.selectbox("Pay Zone", ["Zone A", "Zone B", "Zone C"])

    # Creatinginput data with the same columns used in training
    input_data = pd.DataFrame(columns=model_features)
    input_data.loc[0] = 0

    # Fill numeric variables
    if "Age" in input_data.columns:
        input_data["Age"] = age

    if "Current Employee Rating" in input_data.columns:
        input_data["Current Employee Rating"] = current_rating

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

    if st.button("Predict Performance"):

        if selected_model == "Random Forest":
            prediction = model_RF.predict(input_data)[0]

        elif selected_model == "Logistic Regression":
            input_scaled = scaler.transform(input_data)
            prediction = model_MLR.predict(input_scaled)[0]

        else:
            input_scaled = scaler.transform(input_data)
            prediction = model_SVMC.predict(input_scaled)[0]

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

    st.subheader("Model Performance Table")
    st.dataframe(model_comparison)

    if "Accuracy" in model_comparison.columns:
        st.subheader("Accuracy Comparison")
        st.bar_chart(model_comparison.set_index("Model")["Accuracy"])

    if "F1_Macro" in model_comparison.columns:
        st.subheader("F1 Macro Comparison")
        st.bar_chart(model_comparison.set_index("Model")["F1_Macro"])

    if "F1_Weighted" in model_comparison.columns:
        st.subheader("F1 Weighted Comparison")
        st.bar_chart(model_comparison.set_index("Model")["F1_Weighted"])

#========================================================================
# 3. FAIRNESS ANALYSIS
#========================================================================

elif section == "Fairness Analysis":

    st.header("Fairness Analysis")

    st.write(
        "This section evaluates whether the selected model performs differently "
        "across gender and age groups."
    )

    st.subheader("Gender Fairness")
    st.dataframe(gender_fairness)

    if "F1_Macro" in gender_fairness.columns:
        st.subheader("Gender Fairness - F1 Macro")
        st.bar_chart(gender_fairness.set_index("Gender")["F1_Macro"])

    st.subheader("Age Fairness")
    st.dataframe(age_fairness)

    if "F1_Macro" in age_fairness.columns:
        st.subheader("Age Fairness - F1 Macro")
        st.bar_chart(age_fairness.set_index("Age_Group")["F1_Macro"])
