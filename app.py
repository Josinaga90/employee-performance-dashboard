
import streamlit as st
import joblib
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os


st.set_page_config(page_title="Employee Performance Prediction Dashboard", layout="wide")

#Setting Colour
st.markdown("""
<style>

/* Fondo principal */
.stApp {background-color: #0B1D2A;}

/* Sidebar */
section[data-testid="stSidebar"] {background-color: #081520;}

/* Títulos */
h1, h2, h3 {color: #E6F7FF;}

/* Cards tipo dashboard */
div[data-testid="stMetric"] {
    background-color: #132F44;
    padding: 15px;
    border-radius: 10px;}

/* Botones */
.stButton>button {
    background-color: #00C2D1;
    color: white;
    border-radius: 8px;}

/* Hover botones */
.stButton>button:hover {background-color: #00A6B5;}

/* Dataframe */
.css-1d391kg {background-color: #132F44;}

/* Labels */
label {color: #E6F7FF;}

/* Texto general */
body {color: white;}

</style>
""", unsafe_allow_html=True)



# ================================
# LOAD MODELS
# ================================
@st.cache_resource
def load_data():
    model_RF = joblib.load("model_RF.pkl")
    model_LR = joblib.load("model_LR.pkl")
    model_SVMC = joblib.load("model_SVMC.pkl")
    scaler = joblib.load("scaler.pkl")
    model_features = joblib.load("model_features.pkl")

    return model_RF, model_MLR, model_SVMC, scaler, list(model_features)

#Loading result tables
@st.cache_data
def load_tables():
    employees = pd.read_csv("employees_clean.csv")
    model_comparison_MB = pd.read_csv("model_comparison_modelbase.csv")
    model_comparison_tuned = pd.read_csv("model_comparison_modeltuned.csv")
    age_fairness = pd.read_csv("age_fairness.csv")
    gender_fairness = pd.read_csv("gender_fairness.csv")

    rf_importance = pd.read_csv("rf_importance.csv")
    lr_importance = pd.read_csv("lr_importance.csv")
    svm_importance = pd.read_csv("svm_importance.csv")

    return model_comparison_MB, model_comparison_tuned, age_fairness, gender_fairness, employees, rf_importance, lr_importance, svm_importance

model_RF, model_MLR, model_SVMC, scaler, model_features = load_data()

st.write("RF expected features:", model_RF.n_features_in_)
st.write("LR expected features:", model_MLR.n_features_in_)
st.write("SVM expected features:", model_SVMC.n_features_in_)
st.write("Scaler expected features:", scaler.n_features_in_)
st.write("model_features length:", len(model_features))

model_comparison_MB, model_comparison_tuned, age_fairness, gender_fairness, employees, rf_importance, lr_importance, svm_importance = load_tables()

# Dashboard title
st.title("Employee Performance Prediction Dashboard")

st.write("This dashboard presents individual employee performance prediction, "
    "model comparison, and fairness analysis by gender and age.")

#Sidebar: Create interactive inputs for the HR Manager
section = st.sidebar.selectbox(
    "Select Section",
    ["Individual Prediction", "Dashboard"])

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
    department = st.selectbox("Department Type", ["Production", "Sales", "IT/IS", "Executive Office", "Software Engineering"])
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

        elif selected_model == "Support Vector Machine":
            input_scaled = scaler.transform(input_data)
            prediction = model_SVMC.predict(input_scaled)[0]

        performance_labels = {
            0: "PIP",
            1: "Needs Improvement",
            2: "Fully Meets",
            3: "Exceeds"}

        st.subheader("Prediction Result")
        st.success("Predicted Performance: " + performance_labels[prediction])

# =========================================================
# 2. DASHBOARD
# =========================================================
else:

    st.title("Performance Dashboard")

# ================= KPI =================
    st.subheader("Key Metrics")

    col1, col2, col3 = st.columns(3)

    if "F1_Macro" in model_comparison_tuned.columns:
        best_macro = model_comparison_tuned.loc[model_comparison_tuned["F1_Macro"].idxmax()]
        col1.metric("Best Macro", best_macro["Model"], f"{best_macro['F1_Macro']:.2%}")

    if "F1_Weighted" in model_comparison_tuned.columns:
        best_weighted = model_comparison_tuned.loc[model_comparison_tuned["F1_Weighted"].idxmax()]
        col2.metric("Best Weighted", best_weighted["Model"], f"{best_weighted['F1_Weighted']:.2%}")

    if "Accuracy" in model_comparison_tuned.columns:
        best_acc = model_comparison_tuned.loc[model_comparison_tuned["Accuracy"].idxmax()]
        col3.metric("Best Accuracy", best_acc["Model"], f"{best_acc['Accuracy']:.2%}")

    st.divider()

# ================= DATA INSIGHTS =================
    st.subheader("Data Insights")

    eda = employees.copy()

# ================= FIX VARIABLES =================

    # Target
    eda["Performance"] = eda["Target_Performance"]

    # Gender
    eda["Gender"] = eda["GenderCode"].map({0: "Male", 1: "Female"})

    # Department (reconstrucción desde dummies)
    dept_cols = [col for col in eda.columns if "DepartmentType_" in col]

    def get_department(row):
        for col in dept_cols:
            if row[col] == 1:
                return col.replace("DepartmentType_", "")
        return "Other"

    eda["Department"] = eda.apply(get_department, axis=1)

# ================= GRAPHS =================
    col1, col2 = st.columns(2)

    # Performance Distribution
    fig, ax = plt.subplots(figsize=(3.5, 2))
    sns.countplot(data=eda, x="Performance", palette="Blues", ax=ax)

    ax.set_title("Performance Distribution", color="white", fontsize=10)
    ax.set_facecolor("#0B1D2A")
    fig.patch.set_facecolor("#0B1D2A")
    ax.tick_params(colors="white", labelsize=7)
    ax.set_xlabel("")
    ax.set_ylabel("")
    plt.xticks(rotation=30)
    plt.tight_layout()

    col1.pyplot(fig, use_container_width=False)

    # Performance by Gender
    fig1, ax1 = plt.subplots(figsize=(3.5, 2))
    sns.countplot(data=eda, x="Performance", hue="Gender", ax=ax1)

    ax1.set_title("Performance by Gender", color="white", fontsize=10)
    ax1.set_facecolor("#0B1D2A")
    fig1.patch.set_facecolor("#0B1D2A")
    ax1.tick_params(colors="white", labelsize=7)
    ax1.set_xlabel("")
    ax1.set_ylabel("")
    plt.xticks(rotation=30)
    plt.tight_layout()

    col2.pyplot(fig1, use_container_width=False)

    # Performance by Department
    fig2, ax2 = plt.subplots(figsize=(5, 3))
    sns.countplot(data=eda, x="Department", hue="Performance", ax=ax2)

    ax2.set_title("Performance by Department", color="white", fontsize=10)
    ax2.set_facecolor("#0B1D2A")
    fig2.patch.set_facecolor("#0B1D2A")
    ax2.tick_params(colors="white", labelsize=7)
    ax2.set_xlabel("")
    ax2.set_ylabel("")
    plt.xticks(rotation=30)
    plt.tight_layout()

    st.pyplot(fig2, use_container_width=False)

# ================= MODEL COMPARISON =================
    st.subheader("Model Comparison")

    col1, col2 = st.columns(2)

    col1.write("Base Models")
    col1.dataframe(model_comparison_MB.round(4))

    col2.write("Tuned Models")
    col2.dataframe(model_comparison_tuned.round(4))

    st.divider()

# ================= PERFORMANCE GRAPHS =================
    st.subheader("Performance Comparison")

    col1, col2 = st.columns(2)

    # F1 Macro
    fig4, ax4 = plt.subplots(figsize=(3.5,2))
    sns.barplot(data=model_comparison_tuned, x="Model", y="F1_Macro", palette="Blues")
    ax4.set_title("F1 Macro", color="white", fontsize=10)
    ax4.set_facecolor("#0B1D2A")
    fig4.patch.set_facecolor("#0B1D2A")
    ax4.tick_params(colors="white", labelsize=7)
    col1.pyplot(fig4, use_container_width=False)

    # Accuracy
    fig5, ax5 = plt.subplots(figsize=(3.5,2))
    sns.barplot(data=model_comparison_tuned, x="Model", y="Accuracy", palette="light:cyan")
    ax5.set_title("Accuracy", color="white", fontsize=10)
    ax5.set_facecolor("#0B1D2A")
    fig5.patch.set_facecolor("#0B1D2A")
    ax5.tick_params(colors="white", labelsize=7)
    col2.pyplot(fig5, use_container_width=False)

    st.divider()

# ================= EXPLAINABILITY =================
    st.subheader("Model Explainability")

    col1, col2 = st.columns(2)

    # RF
    with col1:
        fig6, ax6 = plt.subplots(figsize=(4,2.5))
        sns.barplot(data=rf_importance, x="Importance", y="Feature", palette="Blues_r")
        ax6.set_title("Random Forest", color="white", fontsize=10)
        ax6.set_facecolor("#0B1D2A")
        fig6.patch.set_facecolor("#0B1D2A")
        ax6.tick_params(colors="white", labelsize=7)
        st.pyplot(fig6, use_container_width=False)

    # LR
    with col2:
        fig7, ax7 = plt.subplots(figsize=(4,2.5))
        sns.barplot(data=lr_importance, x="Importance", y="Feature", palette="light:cyan")
        ax7.set_title("Logistic Regression", color="white", fontsize=10)
        ax7.set_facecolor("#0B1D2A")
        fig7.patch.set_facecolor("#0B1D2A")
        ax7.tick_params(colors="white", labelsize=7)
        st.pyplot(fig7, use_container_width=False)

    # SVM centrado
    col3, col4, col5 = st.columns([1,2,1])

    with col4:
        fig8, ax8 = plt.subplots(figsize=(5,2.5))
        sns.barplot(data=svm_importance, x="Importance", y="Feature", palette="mako")
        ax8.set_title("Support Vector Machine", color="white", fontsize=10)
        ax8.set_facecolor("#0B1D2A")
        fig8.patch.set_facecolor("#0B1D2A")
        ax8.tick_params(colors="white", labelsize=7)
        st.pyplot(fig8, use_container_width=False)

    st.divider()

# ================= FAIRNESS =================
    st.subheader("Fairness Analysis")

    col1, col2 = st.columns(2)

    col1.write("Gender")
    col1.dataframe(gender_fairness)

    col2.write("Age")
    col2.dataframe(age_fairness)
