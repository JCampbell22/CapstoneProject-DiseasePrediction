

Disease Prediction Using Machine Learning

Overview
This project utilizes machine learning models to predict the risk of various diseases based on input health metrics. The primary focus of this application is on predicting the risk of diabetes, heart disease, and Parkinson's disease. The application is built using Streamlit, a popular framework for creating interactive web applications.

Features
Diabetes Risk Prediction: Predicts the likelihood of diabetes based on input health parameters.
Heart Disease Risk Prediction: Estimates the risk of heart disease.
Parkinson’s Disease Risk Prediction: Assesses the risk of Parkinson’s disease.
Interactive Web Interface: Built with Streamlit for an interactive user experience.

Getting Started

Prerequisites
Python 3.7 or higher
Pip (Python package installer)

Installation
Clone the Repository
git clone https://github.com/yourusername/CapstoneProject-DiseasePrediction.git
cd CapstoneProject-DiseasePrediction

Create a Virtual Environment
python -m venv venv
or
conda create -p venv python==3.8 -y

Activate virtual environment

On windows
venv\Scripts\activate
or
conda activate venv/

On macOS/Linux:
source venv/bin/activate

Install Dependencies
pip install -r requirements.txt

Run the Application
streamlit run app.py
Open your browser and navigate to http://localhost:8501 to use the application.

Models
Diabetes Model: A machine learning model trained to predict diabetes risk.
Heart Disease Model: A model for predicting the risk of heart disease.
Parkinson's Disease Model: A model for assessing the risk of Parkinson's disease.
These models are loaded from saved files and used to make predictions based on user input.

Data
The models use datasets specific to each disease. Ensure that the required dataset files are in the correct directory structure as specified in the app.py script.

Contributing
Contributions are welcome! Please follow these steps:

Fork the repository.
Create a new branch (git checkout -b feature-branch).
Make your changes and commit them (git commit -am 'Add new feature').
Push to the branch (git push origin feature-branch).
Create a new Pull Request.
License
This project is licensed under the MIT License. See the LICENSE file for details.

Contact
For any questions or feedback, please contact:
chhetribinita034@gmail.com

