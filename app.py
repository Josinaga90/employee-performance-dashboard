
import streamlit as st
import joblib
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os


st.set_page_config(page_title="Employee Performance Prediction Dashboard", layout="wide")

# ================================
# DASHBOARD STYLE
# ================================
st.markdown("""
<style>
    .block-container {
        padding-top: 1.2rem;
        padding-bottom: 0rem;
        padding-left: 0.8rem;
        padding-right: 0.8rem;
        max-width: 100%;
    }

    h1 {
        font-size: 32px !important;
        line-height: 1.1 !important;
        margin-top: 0rem !important;
        margin-bottom: 0.6rem !important;
    }

    h2 {
        font-size: 24px !important;
        line-height: 1.1 !important;
        margin-top: 0.2rem !important;
        margin-bottom: 0.4rem !important;
    }

    h3 {
        font-size: 20px !important;
        line-height: 1.1 !important;
        margin-top: 0.2rem !important;
        margin-bottom: 0.4rem !important;
    }

    div[data-testid="stVerticalBlock"] {
        gap: 0.25rem;
    }

    div[data-testid="column"] {
        padding: 0rem 0.15rem;
    }

    div[data-testid="stMetric"] {
        padding: 0rem !important;
    }

    div[data-testid="stMetricLabel"] {
        font-size: 11px !important;
        line-height: 1.1 !important;
    }

    div[data-testid="stMetricValue"] {
        font-size: 20px !important;
        line-height: 1.1 !important;
        white-space: nowrap !important;
    }

    div[data-testid="stMetricDelta"] {
        font-size: 11px !important;
        line-height: 1.1 !important;
    }

    hr {
        margin-top: 0.6rem !important;
        margin-bottom: 0.6rem !important;
    }
</style>
""", unsafe_allow_html=True)

# ================================
# HELPER FUNCTIONS
# ================================

# Function to apply dark style to regular charts
def style_dark_chart(fig, ax, title):

    ax.set_title(title, color="white", fontsize=10, fontweight="bold", pad=5)

    ax.set_facecolor("#061A2E")
    fig.patch.set_facecolor("#061A2E")

    ax.tick_params(colors="white", labelsize=7)

    ax.xaxis.label.set_color("white")
    ax.yaxis.label.set_color("white")

    ax.set_xlabel("")
    ax.set_ylabel("")

    ax.grid(axis="y", color="white", alpha=0.12)
    ax.grid(axis="x", color="white", alpha=0.12)

    for spine in ax.spines.values():
        spine.set_color("#061A2E")

    fig.subplots_adjust(left=0.18, right=0.98, top=0.86, bottom=0.20)


# Function to apply dark style to importance charts
def style_importance_chart(fig, ax, title):

    ax.set_title(title, color="white", fontsize=10, fontweight="bold", pad=5)

    ax.set_facecolor("#061A2E")
    fig.patch.set_facecolor("#061A2E")

    ax.tick_params(colors="white", labelsize=7)

    ax.xaxis.label.set_color("white")
    ax.yaxis.label.set_color("white")

    ax.set_xlabel("")
    ax.set_ylabel("")

    ax.grid(axis="x", color="white", alpha=0.18)
    ax.grid(axis="y", alpha=0)

    for spine in ax.spines.values():
        spine.set_color("#061A2E")

    fig.subplots_adjust(left=0.38, right=0.98, top=0.86, bottom=0.12)


# Function to style legends
def style_legend(ax):

    legend = ax.legend(fontsize=7, title_fontsize=8)

    if legend is not None:
        legend.get_frame().set_facecolor("#061A2E")
        legend.get_frame().set_edgecolor("#061A2E")

        for text in legend.get_texts():
            text.set_color("white")

        if legend.get_title() is not None:
            legend.get_title().set_color("white")


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

    return model_RF, model_LR, model_SVMC, scaler, list(model_features)

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

model_RF, model_LR, model_SVMC, scaler, model_features = load_data()
model_comparison_MB, model_comparison_tuned, age_fairness, gender_fairness, employees, rf_importance, lr_importance, svm_importance = load_tables()

# Dashboard title
st.title("Employee Performance Prediction Dashboard")

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
            prediction = model_LR.predict(input_scaled)[0]

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

    # ================= KPI =================

    st.subheader("Key Metrics")

    col1, col2, col3 = st.columns(3, gap="small")

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

    # Department reconstruction from dummy variables
    dept_cols = [col for col in eda.columns if "DepartmentType_" in col]

    def get_department(row):
        for col in dept_cols:
            if row[col] == 1:
                return col.replace("DepartmentType_", "")
        return "Other"

    eda["Department"] = eda.apply(get_department, axis=1)


    # ================= GRAPHS =================

    col1, col2 = st.columns(2, gap="small")

    # Performance Distribution
    fig, ax = plt.subplots(figsize=(4.6, 2.4))

    sns.countplot(
        data=eda,
        x="Performance",
        hue="Performance",
        palette="crest",
        legend=False,
        ax=ax)

    style_dark_chart(fig, ax, "Performance Distribution")
    plt.xticks(rotation=25)
    col1.pyplot(fig, use_container_width=True)


    # Performance by Gender
    fig1, ax1 = plt.subplots(figsize=(4.6, 2.4))

    sns.countplot(
        data=eda,
        x="Performance",
        hue="Gender",
        palette="crest",
        ax=ax1)

    style_dark_chart(fig1, ax1, "Performance by Gender")
    style_legend(ax1)
    plt.xticks(rotation=25)
    col2.pyplot(fig1, use_container_width=True)


    # Performance by Department
    fig2, ax2 = plt.subplots(figsize=(9.6, 2.8))

    sns.countplot(
        data=eda,
        x="Department",
        hue="Performance",
        palette="crest",
        ax=ax2)

    style_dark_chart(fig2, ax2, "Performance by Department")
    style_legend(ax2)
    plt.xticks(rotation=25)
    st.pyplot(fig2, use_container_width=True)

    st.divider()


    # ================= MODEL COMPARISON =================

    st.subheader("Model Comparison")

    col1, col2 = st.columns(2, gap="small")

    col1.write("Base Models")
    col1.dataframe(model_comparison_MB.round(4), use_container_width=True)

    col2.write("Tuned Models")
    col2.dataframe(model_comparison_tuned.round(4), use_container_width=True)

    st.divider()


    # ================= PERFORMANCE GRAPHS =================

    st.subheader("Performance Comparison")

    col1, col2 = st.columns(2, gap="small")

    # F1 Macro
    fig4, ax4 = plt.subplots(figsize=(4.6, 2.4))

    sns.barplot(
        data=model_comparison_tuned,
        x="Model",
        y="F1_Macro",
        hue="Model",
        palette="crest",
        legend=False,
        ax=ax4)

    style_dark_chart(fig4, ax4, "F1 Macro")
    plt.xticks(rotation=20)
    col1.pyplot(fig4, use_container_width=True)


    # Accuracy
    fig5, ax5 = plt.subplots(figsize=(4.6, 2.4))

    sns.barplot(
        data=model_comparison_tuned,
        x="Model",
        y="Accuracy",
        hue="Model",
        palette="crest",
        legend=False,
        ax=ax5)

    style_dark_chart(fig5, ax5, "Accuracy")
    plt.xticks(rotation=20)
    col2.pyplot(fig5, use_container_width=True)

    st.divider()


    # ================= EXPLAINABILITY =================

    st.subheader("Model Explainability")

    col1, col2, col3 = st.columns(3, gap="small")

    # Random Forest Importance
    with col1:

        fig6, ax6 = plt.subplots(figsize=(4.3, 2.7))

        sns.barplot(
            data=rf_importance,
            x="Importance",
            y="Feature",
            hue="Feature",
            palette="crest",
            legend=False,
            ax=ax6)

        style_importance_chart(fig6, ax6, "Random Forest")
        st.pyplot(fig6, use_container_width=True)


    # Logistic Regression Importance
    with col2:

        fig7, ax7 = plt.subplots(figsize=(4.3, 2.7))

        sns.barplot(
            data=lr_importance,
            x="Importance",
            y="Feature",
            hue="Feature",
            palette="crest",
            legend=False,
            ax=ax7)

        style_importance_chart(fig7, ax7, "Logistic Regression")
        st.pyplot(fig7, use_container_width=True)


    # Support Vector Machine Importance
    with col3:

        fig8, ax8 = plt.subplots(figsize=(4.3, 2.7))

        sns.barplot(
            data=svm_importance,
            x="Importance",
            y="Feature",
            hue="Feature",
            palette="crest",
            legend=False,
            ax=ax8)

        style_importance_chart(fig8, ax8, "Support Vector Machine")
        st.pyplot(fig8, use_container_width=True)

    st.divider()


    # ================= FAIRNESS =================

    st.subheader("Fairness Analysis")

    col1, col2 = st.columns(2, gap="small")

    col1.write("Gender")
    col1.dataframe(gender_fairness, use_container_width=True)

    col2.write("Age")
    col2.dataframe(age_fairness, use_container_width=True)
