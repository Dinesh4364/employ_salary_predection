import streamlit as st
import pandas as pd
import pickle
import numpy as np

try:
    with open('onehot_encoder.pkl', 'rb') as f:
        encoder = pickle.load(f)
except FileNotFoundError:
    st.error("Error: 'onehot_encoder.pkl' not found. Please ensure it's in the same directory.")
    st.stop()

# Load the pre-trained model
try:
    with open('gradient_boosting_model.pkl', 'rb') as f:
        model = pickle.load(f)
except FileNotFoundError:
    st.error("Error: 'gradient_boosting_model.pkl' not found. Please ensure it's in the same directory.")
    st.stop()

def preprocess_input(input_df, encoder):
   
    if 'workclass' in input_df.columns:
        input_df['workclass'] = input_df['workclass'].replace('?', 'unknow')
    if 'occupation' in input_df.columns:
        input_df['occupation'] = input_df['occupation'].replace('?', 'unknow')
    if 'native-country' in input_df.columns:
        input_df['native-country'] = input_df['native-country'].replace('?', 'unknow')
        
    occupation_rare_mapping = {
        'Priv-house-serv': 'Other', 'Armed-Forces': 'Other',
        'Handlers-cleaners': 'Other', 'Transport-moving': 'Other',
        'Farming-fishing': 'Other', 'Machine-op-inspct': 'Other',
        'Protective-serv': 'Other', 'Other-service': 'Other',
        'unknow': 'Other' 
    }
    if 'occupation' in input_df.columns:
        input_df['occupation'] = input_df['occupation'].replace(occupation_rare_mapping)
       
    country_rare_mapping = {
        'Japan': 'Other', 'South': 'Other', 'Cuba': 'Other', 'Iran': 'Other',
        'Italy': 'Other', 'Poland': 'Other', 'Jamaica': 'Other', 'France': 'Other',
        'Vietnam': 'Other', 'Puerto-Rico': 'Other', 'El-Salvador': 'Other',
        'Columbia': 'Other', 'Hong': 'Other', 'Thailand': 'Other', 'Peru': 'Other',
        'Haiti': 'Other', 'Greece': 'Other', 'Ireland': 'Other',
        'Dominican-Republic': 'Other', 'Hungary': 'Other', 'Ecuador': 'Other',
        'Nicaragua': 'Other', 'Scotland': 'Other', 'Outlying-US(Guam-USVI-etc)': 'Other',
        'Yugoslavia': 'Other', 'Cambodia': 'Other', 'Laos': 'Other',
        'Trinadad&Tobago': 'Other', 'Guatemala': 'Other', 'Portugal': 'Other',
        'Honduras': 'Other', 'unknow': 'Other' # This was explicitly done in the notebook
    }
    if 'native-country' in input_df.columns:
        input_df['native-country'] = input_df['native-country'].replace(country_rare_mapping)

    if 'gender' in input_df.columns:
        input_df['gender'] = input_df['gender'].replace({'Male': 0, 'Female': 1})

    cat_cols = ['occupation', 'native-country', 'workclass']

    encoded_array = encoder.transform(input_df[cat_cols])
    encoded_df = pd.DataFrame(encoded_array, columns=encoder.get_feature_names_out(cat_cols), index=input_df.index)

    input_df = input_df.drop(columns=cat_cols)
    input_df = pd.concat([input_df, encoded_df], axis=1)

    # Convert all boolean columns to int (if any were created by encoding)
    bool_cols = input_df.select_dtypes(include=['bool']).columns
    input_df[bool_cols] = input_df[bool_cols].astype(int)

    expected_columns = ['age', 'educational-num', 'gender', 'hours-per-week'] + list(encoder.get_feature_names_out(cat_cols))

    # Reindex the input_df to match the expected columns, filling missing with 0
    processed_df = input_df.reindex(columns=expected_columns, fill_value=0)

    return processed_df

# Streamlit App
st.title("Employee Salary Prediction")
st.write("Predict if an employee's income is >50K or <=50K based on their attributes.")

st.header("Enter Employee Details:")

# Input fields
age = st.slider("Age", 17, 90, 30)
workclass = st.selectbox("Workclass", ['Private', 'Self-emp-not-inc', 'Local-gov', 'Federal-gov', 'State-gov', 'Self-emp-inc', 'Without-pay', 'Never-worked', 'unknow'])
educational_num = st.slider("Educational Num (Higher is more education)", 12, 16, 13) # Filtered >=12 in notebook
occupation = st.selectbox("Occupation", [
    'Prof-specialty', 'Exec-managerial', 'Sales', 'Adm-clerical',
    'Craft-repair', 'Tech-support', 'Other', 'Protective-serv',
    'Machine-op-inspct', 'Farming-fishing', 'Transport-moving',
    'Handlers-cleaners', 'Priv-house-serv', 'Armed-Forces', 'unknow'
])
gender = st.selectbox("Gender", ['Male', 'Female'])
hours_per_week = st.slider("Hours per Week", 1, 99, 40)
native_country = st.selectbox("Native Country", [
    'United-States', 'Mexico', 'Philippines', 'Germany', 'Puerto-Rico',
    'Canada', 'El-Salvador', 'India', 'Cuba', 'England', 'China', 'South',
    'Jamaica', 'Italy', 'Dominican-Republic', 'Japan', 'Guatemala',
    'Poland', 'Vietnam', 'Columbia', 'Haiti', 'Portugal', 'Taiwan', 'Iran',
    'Nicaragua', 'Greece', 'Peru', 'Ecuador', 'France', 'Ireland', 'Thailand',
    'Hong', 'Cambodia', 'Trinadad&Tobago', 'Laos', 'Outlying-US(Guam-USVI-etc)',
    'Yugoslavia', 'Scotland', 'Honduras', 'Hungary', 'Holand-Netherlands', 'unknow'
])

# Create a DataFrame from user input
input_data = pd.DataFrame([{
    'age': age,
    'workclass': workclass,
    'educational-num': educational_num,
    'occupation': occupation,
    'gender': gender,
    'hours-per-week': hours_per_week,
    'native-country': native_country
}])

if st.button("Predict Income"):
    # Preprocess the input data
    processed_input = preprocess_input(input_data.copy(), encoder)

    # Make prediction
    prediction = model.predict(processed_input)
    prediction_proba = model.predict_proba(processed_input)

    st.subheader("Prediction Result:")
    if prediction[0] == 1:
        st.success(f"The predicted income is **>50K**")
    else:
        st.info(f"The predicted income is **<=50K**")

 
