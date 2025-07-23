# Employee Salary Prediction

## Project Overview
This project predicts whether an employee's annual salary exceeds $50K based on demographic and occupational features. Using machine learning techniques, specifically a Gradient Boosting Classifier, the model classifies income levels to assist in HR decision-making and recruitment processes.

## Features
- Handles missing data by replacing unknown values.
- Groups rare categories for better model performance.
- Encodes categorical variables using OneHotEncoder.
- Predicts salary class using Gradient Boosting Classifier.
- Provides an interactive web interface via Streamlit for real-time predictions.

## System Requirements
- Operating System: Windows, macOS, or Linux
- Python 3.x
- RAM: Minimum 4 GB (8 GB recommended)
- At least 1 GB free disk space

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/employee-salary-prediction.git
   cd employee-salary-prediction
Create and activate a virtual environment (optional but recommended):

bash
Copy
Edit
python -m venv env
source env/bin/activate   # On Windows: env\Scripts\activate
Install required libraries:

bash
Copy
Edit
pip install -r requirements.txt
Usage
Training the Model
Open the Jupyter notebook Employee_Salary_Prediction.ipynb.

Run all cells to preprocess data, train the model, and save the trained model and encoder.

Running the Streamlit App
Run the Streamlit app:

bash
Copy
Edit
streamlit run app.py
Use the web interface to input employee details and get salary predictions in real-time.

Project Structure
bash
Copy
Edit
├── Employee_Salary_Prediction.ipynb  # Jupyter notebook for data processing and model training
├── app.py                            # Streamlit app for real-time salary prediction
├── Project template 4.pptx           # Presentation template used
├── model.pkl                        # Saved trained model (after running notebook)
├── encoder.pkl                      # Saved encoder (after running notebook)
├── requirements.txt                 # Required Python packages
└── README.md                       # Project overview and instructions
Libraries Used
pandas

numpy

scikit-learn

pickle

streamlit

Future Enhancements
Incorporate additional features like education level and marital status.

Experiment with other machine learning algorithms to improve accuracy.

Deploy as an API for wider enterprise use.

Address model fairness and bias considerations.
