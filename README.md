# Project README: Multiple Disease Prediction

## Overview

This project aims to develop a predictive model for various diseases using machine learning techniques. The dataset includes health-related features that can help in predicting the likelihood of diseases such as diabetes and heart disease. The project utilizes Python libraries such as Streamlit for the web application interface and Scikit-learn for machine learning algorithms.

## Project Structure

The project consists of the following key files:

- **`requirements.txt`**: Contains the list of dependencies required to run the project.
- **`diabetes.csv`**: A dataset containing health metrics related to diabetes.
- **`heart.csv`**: A dataset containing health metrics related to heart disease.
- **`multiplediseaseprediction.py`**: The main Python script that implements the machine learning model and the web application.

## Installation

To set up the project, follow these steps:

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd <repository-directory>
   ```

2. **Create a virtual environment** (optional but recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   ```

3. **Install the required packages**:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

To run the application, execute the following command:

```bash
streamlit run multiplediseaseprediction.py
```

This will start a local web server, and you can access the application in your web browser at `http://localhost:8501`.

## Datasets

### Diabetes Dataset (`diabetes.csv`)

This dataset contains the following columns:

- **Pregnancies**: Number of times pregnant
- **Glucose**: Plasma glucose concentration a 2 hours in an oral glucose tolerance test
- **BloodPressure**: Diastolic blood pressure (mm Hg)
- **SkinThickness**: Triceps skin fold thickness (mm)
- **Insulin**: 2-Hour serum insulin (mu U/ml)
- **BMI**: Body mass index (weight in kg/(height in m)^2)
- **DiabetesPedigreeFunction**: Diabetes pedigree function
- **Age**: Age (years)
- **Outcome**: Class variable (0 or 1)

### Heart Disease Dataset (`heart.csv`)

This dataset contains the following columns:

- **age**: Age of the patient
- **sex**: Sex of the patient (1 = male; 0 = female)
- **cp**: Chest pain type (0-3)
- **trestbps**: Resting blood pressure (in mm Hg)
- **chol**: Serum cholesterol in mg/dl
- **fbs**: Fasting blood sugar > 120 mg/dl (1 = true; 0 = false)
- **restecg**: Resting electrocardiographic results (0-2)
- **thalach**: Maximum heart rate achieved
- **exang**: Exercise induced angina (1 = yes; 0 = no)
- **oldpeak**: ST depression induced by exercise relative to rest
- **slope**: Slope of the peak exercise ST segment (0-2)
- **ca**: Number of major vessels (0-3) colored by fluoroscopy
- **thal**: Thalassemia (1 = normal; 2 = fixed defect; 3 = reversable defect)
- **target**: Diagnosis of heart disease (1 = presence; 0 = absence)

## Contributing

Contributions are welcome! If you have suggestions for improvements or new features, please open an issue or submit a pull request.

## License

This project is licensed under the MIT License. See the LICENSE file for more details.

---

