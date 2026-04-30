
import streamlit as st
import joblib
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.inspection import permutation_importance

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
    model_MLR = joblib.load("model_MLR.pkl")
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

    return model_comparison_MB, model_comparison_tuned, age_fairness, gender_fairness, employees

model_RF, model_MLR, model_SVMC, scaler, model_features = load_data()
model_comparison_MB, model_comparison_tuned, age_fairness, gender_fairness, employees = load_tables()

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

    # ================= TABLES =================
    st.subheader("Model Comparison")

    col1, col2 = st.columns(2)

    col1.write("Base Models")
    col1.dataframe(model_comparison_MB.round(4))

    col2.write("Tuned Models")
    col2.dataframe(model_comparison_tuned.round(4))

    # ================= GRAPHS =================
    st.subheader("Performance Comparison")

    plt.style.use("dark_background")
    sns.set_theme(style="dark")

    # F1 Macro
    fig, ax = plt.subplots()
    sns.barplot(data=model_comparison_tuned,
        x="Model",
        y="F1_Macro",
        palette="Blues")
    ax.set_title("F1 Macro Comparison", color="white", fontsize=12, fontweight="bold")
    ax.set_facecolor("#0B1D2A")
    fig.patch.set_facecolor("#0B1D2A")
    ax.tick_params(axis='x', colors='white', labelsize=8)
    ax.tick_params(axis='y', colors='white', labelsize=8)
    ax.set_xlabel("Model", color="white")
    ax.set_ylabel("Score", color="white")
    
    col1.pyplot(fig)

    # Accuracy
    fig2, ax2 = plt.subplots()
    sns.barplot(data=model_comparison_tuned,
        x="Model",
        y="Accuracy",
        palette="light:cyan")
    ax2.set_title("Accuracy Comparison", color="white", fontsize=12, fontweight="bold")
    ax2.set_facecolor("#0B1D2A")
    fig2.patch.set_facecolor("#0B1D2A")
    ax2.tick_params(axis='x', colors='white', labelsize=8)
    ax2.tick_params(axis='y', colors='white', labelsize=8)
    ax2.set_xlabel("Model", color="white")
    ax2.set_ylabel("Score", color="white")

    col2.pyplot(fig2)

    # ================= DATA INSIGHTS =================
    st.subheader("Data Insights")

    fig3, ax3 = plt.subplots(figsize=(4,2.5))
    sns.countplot(data=employees, x="Performance Score", palette="Blues")
    ax3.set_title("Performance Distribution", color="white")
    ax3.set_facecolor("#0B1D2A")
    fig3.patch.set_facecolor("#0B1D2A")
    ax3.tick_params(colors="white")
    st.pyplot(fig3)

    fig4, ax4 = plt.subplots(figsize=(4,2.5))
    sns.countplot(data=employees, x="Performance Score", hue="GenderCode")
    ax4.set_title("By Gender", color="white")
    ax4.set_facecolor("#0B1D2A")
    fig4.patch.set_facecolor("#0B1D2A")
    ax4.tick_params(colors="white")
    st.pyplot(fig4)

    fig5, ax5 = plt.subplots(figsize=(4,2.5))
    sns.countplot(data=employees, x="DepartmentType", hue="Performance Score")
    ax5.set_title("By Department", color="white")
    plt.xticks(rotation=30)
    ax5.set_facecolor("#0B1D2A")
    fig5.patch.set_facecolor("#0B1D2A")
    ax5.tick_params(colors="white")
    st.pyplot(fig5)

    # ================= EXPLAINABILITY =================
    st.subheader("Model Explainability")

    # RF
    imp = pd.Series(model_RF.feature_importances_, index=model_features).nlargest(6)
    fig6, ax6 = plt.subplots(figsize=(4,2.5))
    sns.barplot(x=imp.values, y=imp.index, palette="Blues_r")
    ax6.set_title("Random Forest Importance", color="white")
    ax6.set_facecolor("#0B1D2A")
    fig6.patch.set_facecolor("#0B1D2A")
    ax6.tick_params(colors="white")
    st.pyplot(fig6)

    # LR
    coef_values = model_MLR.coef_[0]
    coef = pd.Series(coef_values, index=model_features[:len(coef_values)])
    coef = coef.abs().nlargest(6)

    fig7, ax7 = plt.subplots(figsize=(4,2.5))
    sns.barplot(x=coef.values, y=coef.index, palette="light:cyan")
    ax7.set_title("Logistic Regression Importance", color="white", fontsize=10)
    ax7.set_facecolor("#0B1D2A")
    fig7.patch.set_facecolor("#0B1D2A")
    ax7.tick_params(colors="white", labelsize=7)
    plt.tight_layout()

    col2.pyplot(fig7)

    # SVM
    perm = permutation_importance(model_SVMC,
        employees[model_features],
        model_SVMC.predict(employees[model_features]))

    svm_imp = pd.Series(perm.importances_mean, index=model_features).nlargest(6)

    fig8, ax8 = plt.subplots(figsize=(4,2.5))
    sns.barplot(x=svm_imp.values, y=svm_imp.index, palette="mako")
    ax8.set_title("SVM Importance", color="white")
    ax8.set_facecolor("#0B1D2A")
    fig8.patch.set_facecolor("#0B1D2A")
    ax8.tick_params(colors="white")
    st.pyplot(fig8)


    # ================= FAIRNESS =================
    st.subheader("Fairness Analysis")

    col1, col2 = st.columns(2)

    col1.write("Gender")
    col1.dataframe(gender_fairness)

    col2.write("Age")
    col2.dataframe(age_fairness)

    col1, col2 = st.columns(2)

