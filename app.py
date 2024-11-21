import streamlit as st
import pandas as pd
import joblib
import numpy as np
from sklearn.preprocessing import LabelEncoder

# Load your pre-trained models from the directory
model_paths = {
    "Random Forest": "rf_classifier",
    "Logistic Regression": "logreg_classifier",
    "LightGBM": "lgbm_classifier",
    "XGBoost": "xgb_classifier",
}  
models = {name: joblib.load(path) for name, path in model_paths.items()}

# Load the pre-fitted label encoders
encoders = {
    'Device_Name': joblib.load('Device_Name_encoder.pkl'),
    'Attack': joblib.load('Attack_encoder.pkl'),
    'Attack_subType': joblib.load('Attack_subType_encoder.pkl')
}

# Function to predict using the selected model
def predict(model, input_data):
    return model.predict(input_data)

# Streamlit interface
st.title('IoT Intrusion Detection System')

# Create input fields for each column
st.header('Input Features')
input_data = {}

input_data['MI_dir_L0.1_weight'] = st.number_input('MI_dir_L0.1_weight', format="%.6f")
input_data['MI_dir_L0.1_mean'] = st.number_input('MI_dir_L0.1_mean', format="%.6f")
input_data['MI_dir_L0.1_variance'] = st.number_input('MI_dir_L0.1_variance', format="%.6f")
input_data['H_L0.1_weight'] = st.number_input('H_L0.1_weight', format="%.6f")
input_data['H_L0.1_mean'] = st.number_input('H_L0.1_mean', format="%.6f")
input_data['H_L0.1_variance'] = st.number_input('H_L0.1_variance', format="%.6f")
input_data['HH_L0.1_weight'] = st.number_input('HH_L0.1_weight', format="%.6f")
input_data['HH_L0.1_mean'] = st.number_input('HH_L0.1_mean', format="%.6f")
input_data['HH_L0.1_std'] = st.number_input('HH_L0.1_std', format="%.6f")
input_data['HH_L0.1_magnitude'] = st.number_input('HH_L0.1_magnitude', format="%.6f")
input_data['HH_L0.1_radius'] = st.number_input('HH_L0.1_radius', format="%.6f")
input_data['HH_L0.1_covariance'] = st.number_input('HH_L0.1_covariance', format="%.6f")
input_data['HH_L0.1_pcc'] = st.number_input('HH_L0.1_pcc', format="%.6f")
input_data['HH_jit_L0.1_weight'] = st.number_input('HH_jit_L0.1_weight', format="%.6f")
input_data['HH_jit_L0.1_mean'] = st.number_input('HH_jit_L0.1_mean', format="%.6f")
input_data['HH_jit_L0.1_variance'] = st.number_input('HH_jit_L0.1_variance', format="%.6f")
input_data['HpHp_L0.1_weight'] = st.number_input('HpHp_L0.1_weight', format="%.6f")
input_data['HpHp_L0.1_mean'] = st.number_input('HpHp_L0.1_mean', format="%.6f")
input_data['HpHp_L0.1_std'] = st.number_input('HpHp_L0.1_std', format="%.6f")
input_data['HpHp_L0.1_magnitude'] = st.number_input('HpHp_L0.1_magnitude', format="%.6f")
input_data['HpHp_L0.1_radius'] = st.number_input('HpHp_L0.1_radius', format="%.6f")
input_data['HpHp_L0.1_covariance'] = st.number_input('HpHp_L0.1_covariance', format="%.6f")
input_data['HpHp_L0.1_pcc'] = st.number_input('HpHp_L0.1_pcc', format="%.6f")

# Dropdown menu for 'Device_Name'
input_data['Device_Name'] = st.selectbox(
    'Device_Name', 
    [
        'Philips_B120N10_Baby_Monitor', 
        'Danmini_Doorbell', 
        'SimpleHome_XCS7_1002_WHT_Security_Camera', 
        'SimpleHome_XCS7_1003_WHT_Security_Camera', 
        'Provision_PT_838_Security_Camera', 
        'Ecobee_Thermostat', 
        'Provision_PT_737E_Security_Camera', 
        'Samsung_SNH_1011_N_Webcam', 
        'Ennio_Doorbell'
    ]
)

# Dropdown menu for 'Attack'
input_data['Attack'] = st.selectbox(
    'Attack', 
    ['mirai', 'gafgyt', 'Normal']
)

# Dropdown menu for 'Attack_subType'
input_data['Attack_subType'] = st.selectbox(
    'Attack_subType', 
    ['udp', 'tcp', 'scan', 'syn', 'ack', 'Normal', 'udpplain', 'combo', 'junk']
)

# Convert input data to DataFrame
input_df = pd.DataFrame([input_data])

# Apply Label Encoding using the loaded encoders
for column, encoder in encoders.items():
    input_df[column] = encoder.transform(input_df[column])

# Model selection
model_choice = st.selectbox("Choose a model", options=list(models.keys()))
selected_model = models[model_choice]

# Prediction button
if st.button('Predict'):
    # Apply the selected model
    prediction = predict(selected_model, input_df)
    st.write("Prediction:", prediction)

# Output section
st.header('Output')
st.write("Prediction will be displayed here after input and model selection.")
