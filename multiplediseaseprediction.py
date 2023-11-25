import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import pickle

# Load diabetes data
diabetes_data = pd.read_csv("diabetes.csv")

# Load heart disease data
heart_data = pd.read_csv("heart.csv")

# Load diabetes model
with open("diabetes_model.sav", "rb") as model_file:
    diabetes_model = pickle.load(model_file)

# Load heart disease model
with open("heart_disease_model.sav", "rb") as model_file:
    heart_model = pickle.load(model_file)

# Define units for each input
diabetes_data_units = {
    "Pregnancies": "",
    "Glucose": "mg/dl",
    "BloodPressure": "mm Hg",
    "SkinThickness": "mm",
    "Insulin": "mu U/ml",
    "BMI": "kg/m^2",
    "DiabetesPedigreeFunction": "",
    "Age": "years"
}

heart_data_units = {
    "Age": "years",
    "Sex": "",
    "ChestPainType": "",
    "RestingBloodPressure": "mm Hg",
    "Cholesterol": "mg/dl",
    "FastingBloodSugar": "",
    "RestingECG": "",
    "MaxHeartRate": "bpm",
    "ExerciseAngina": "",
    "STDepression": "",
    "Slope": "",
    "MajorVessels": "",
    "Thalassemia": ""
}

# Define normal ranges for each input (these are assumed values, consult with a medical professional)
normal_ranges_diabetes = {
    "Pregnancies": (0, 17),
    "Glucose": (0, 199),
    "BloodPressure": (0, 122),
    "SkinThickness": (0, 99),
    "Insulin": (0, 846),
    "BMI": (0.0, 67.1),
    "DiabetesPedigreeFunction": (0.078, 2.42),
    "Age": (21, 81)
}

normal_ranges_heart = {
    "Age": (29, 77),
    "RestingBloodPressure": (94, 200),
    "Cholesterol": (126, 564),
    "MaxHeartRate": (71, 202),
    "STDepression": (0.0, 6.2)
}

def get_diabetes_prediction(inputs):
    input_df = pd.DataFrame([inputs], columns=diabetes_data.columns[:-1])

    try:
        probability = diabetes_model.predict_proba(input_df)[:, 1]
        prediction = (probability > 0.5).astype(int)
    except AttributeError:
        try:
            decision_function_value = diabetes_model.decision_function(input_df)
            probability = 1 / (1 + np.exp(-decision_function_value))
            prediction = (probability > 0.5).astype(int)
        except AttributeError:
            prediction = diabetes_model.predict(input_df)
            probability = None

    return prediction, probability[0] * 100 if probability is not None else None

def get_heart_disease_prediction(inputs):
    input_df = pd.DataFrame([inputs], columns=heart_data.columns[:-1])

    try:
        probability = heart_model.predict_proba(input_df)[:, 1]
        prediction = (probability > 0.5).astype(int)
    except AttributeError:
        try:
            decision_function_value = heart_model.decision_function(input_df)
            probability = 1 / (1 + np.exp(-decision_function_value))
            prediction = (probability > 0.5).astype(int)
        except AttributeError:
            prediction = heart_model.predict(input_df)
            probability = None

    return prediction, probability[0] * 100 if probability is not None else None

def plot_meter(value, label, normal_range, unit):
    fig, ax = plt.subplots(figsize=(8, 2))
    ax.barh([label], [value], color='skyblue', label='Patient')
    ax.barh([label], [normal_range[1]], color='lightgreen', alpha=0.7, label='Normal Range')
    ax.set_xlim(0, max(value, normal_range[1]) + 20)
    ax.set_xlabel(f'{label} ({unit})')
    ax.legend()
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['left'].set_color('gray')
    ax.spines['bottom'].set_color('gray')
    ax.tick_params(axis='both', colors='gray')
    ax.yaxis.label.set_color('gray')
    st.pyplot(fig)

def main():
    st.set_page_config(
        page_title="Health Prediction App",
        page_icon="chart_with_upwards_trend",
        layout="wide"
    )

    st.title("Health Prediction App")
    st.sidebar.title("Choose the Form to Fill")

    form_choice = st.sidebar.radio(
        "Select Form",
        ('Diabetes', 'Heart Disease')
    )

    if form_choice == 'Diabetes':
        st.header("Diabetes Prediction Form")
        pregnancies = st.slider("Pregnancies (count)", 0, 17, 1)
        glucose = st.slider("Glucose (mg/dl)", 0, 199, 120)
        blood_pressure = st.slider("Blood Pressure (mm Hg)", 0, 122, 70)
        skin_thickness = st.slider("Skin Thickness (mm)", 0, 99, 20)
        insulin = st.slider("Insulin (mu U/ml)", 0, 846, 79)
        bmi = st.slider("BMI (kg/m^2)", 0.0, 67.1, 30.0)
        pedigree_function = st.slider("Diabetes Pedigree Function", 0.078, 2.42, 0.3725)
        age = st.slider("Age (years)", 21, 81, 35)

        inputs = [pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, pedigree_function, age]

        if st.button("Predict Diabetes"):
            prediction, probability = get_diabetes_prediction(inputs)
            st.write(f"Prediction: {'Diabetic' if prediction[0] == 1 else 'Not Diabetic'}")
            st.write(f"Probability: {probability:.2f}%")

            # Display the meter for each input
            for label, value in zip(diabetes_data.columns[:-1], inputs):
                normal_range = normal_ranges_diabetes.get(label, (0, 100))
                unit = diabetes_data_units.get(label, '')
                plot_meter(value, label, normal_range, unit)

    elif form_choice == 'Heart Disease':
        st.header("Heart Disease Prediction Form")
        age = st.slider("Age (years)", 29, 77, 40)
        sex = st.radio("Sex", ("Male", "Female"))
        sex = 1 if sex == "Male" else 0
        cp = st.slider("Chest Pain Type", 0, 3, 1)
        trestbps = st.slider("Resting Blood Pressure (mm Hg)", 94, 200, 120)
        chol = st.slider("Cholesterol (mg/dl)", 126, 564, 200)
        fbs = st.radio("Fasting Blood Sugar", ("<= 120 mg/dl", "> 120 mg/dl"))
        fbs = 1 if fbs == "> 120 mg/dl" else 0
        restecg = st.slider("Resting Electrocardiographic Results", 0, 2, 1)
        thalach = st.slider("Maximum Heart Rate Achieved (bpm)", 71, 202, 150)
        exang = st.radio("Exercise Induced Angina", ("Yes", "No"))
        exang = 1 if exang == "Yes" else 0
        oldpeak = st.slider("ST Depression Induced by Exercise Relative to Rest", 0.0, 6.2, 0.0)
        slope = st.slider("Slope of the Peak Exercise ST Segment", 0, 2, 1)
        ca = st.slider("Number of Major Vessels Colored by Fluoroscopy", 0, 4, 0)
        thal = st.slider("Thalassemia", 0, 3, 2)

        inputs = [age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal]

        if st.button("Predict Heart Disease"):
            prediction, probability = get_heart_disease_prediction(inputs)
            st.write(f"Prediction: {'Heart Disease' if prediction[0] == 1 else 'No Heart Disease'}")
            st.write(f"Probability: {probability:.2f}%")

            # Display the meter for each input
            for label, value in zip(heart_data.columns[:-1], inputs):
                normal_range = normal_ranges_heart.get(label, (0, 100))
                unit = heart_data_units.get(label, '')
                plot_meter(value, label, normal_range, unit)
    
if __name__ == "__main__":
    main()
