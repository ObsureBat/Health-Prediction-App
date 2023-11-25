Health Prediction App

Overview
The Health Prediction App is a web-based application built using Streamlit that predicts the likelihood of two health conditions: Diabetes and Heart Disease. The app leverages machine learning models trained on relevant datasets to provide users with predictions based on input parameters related to health metrics.

Features
Diabetes Prediction:

Users can input parameters such as pregnancies, glucose levels, blood pressure, and more to predict the likelihood of diabetes.
Heart Disease Prediction:

Users can input parameters including age, gender, chest pain type, cholesterol levels, and more to predict the likelihood of heart disease.
Visualizations:

The app includes interactive visualizations for each input parameter, providing users with a clear understanding of their health metrics in comparison to normal ranges.
Usage
Select Form:

Choose between the 'Diabetes' and 'Heart Disease' prediction forms in the sidebar.
Input Parameters:

Fill in the relevant health parameters using sliders, radio buttons, and other interactive elements.
Prediction:

Click the "Predict Diabetes" or "Predict Heart Disease" button to obtain predictions based on the entered parameters.
Visualizations:

Visual representations of input parameters are displayed in meters, comparing user values to normal ranges.
Installation
Clone the repository:

bash
Copy code
git clone https://github.com/your-username/health-prediction-app.git
Install the required dependencies:

bash
Copy code
pip install -r requirements.txt
Run the app:

bash
Copy code
streamlit run app.py
Dependencies
Streamlit
Pandas
Matplotlib
NumPy
Scikit-learn
Models
The app utilizes pre-trained machine learning models for diabetes and heart disease prediction. Model files are stored in the repository.
Contributing
Contributions are welcome! Feel free to open issues or submit pull requests for any improvements, bug fixes, or new features.

License
This project is licensed under the MIT License.
