import streamlit as st
import numpy as np
import joblib

# ----------------- LOGIN SETUP -----------------
USERNAME = "muskan"
PASSWORD = "Muskan@2025"

if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

if not st.session_state.logged_in:
    st.title("🔐 Login to Access Dashboard")

    username = st.text_input("Username")
    password = st.text_input("Password", type="password")
    login_btn = st.button("Login")

    if login_btn:
        if username == USERNAME and password == PASSWORD:
            st.session_state.logged_in = True
            st.rerun()
        else:
            st.error("❌ Invalid credentials")

    st.stop()
# Load saved model
model = joblib.load("model_top12.pkl")

# ----------------------------
# Normalization helper
def normalize_input(value, min_val, max_val):
    if max_val == min_val:
        return 0
    return (value - min_val) / (max_val - min_val)

# ----------------------------
# Min-max dictionary for other features (NOT for BMI, Age, Weight)
min_max_dict = {
    'Medical_History_4': (1.0, 2.0),
    'Product_Info_4': (0.0, 1.0),
    'InsuredInfo_6': (1.0, 2.0),
    'Family_Hist_1': (1.0, 3.0),
    'Insurance_History_9': (1.0, 3.0),
    'Product_Info_2': (0.0, 18.0),
    'Medical_History_1': (0.0, 17.0),
    'Insurance_History_3': (1.0, 3.0),
    'Family_Hist_4': (0.211268, 0.661972)
}

# ----------------------------
# Streamlit UI
st.title("🩺 Life Insurance Risk Predictor")
st.markdown("Enter applicant details to predict the **Response Level (1 to 8)**")

# Inject custom CSS for Predict button
st.markdown("""
    <style>
    div.stButton > button:first-child {
        background-color: #28a745;
        color: white;
        width: 200px;
        height: 45px;
        font-size: 16px;
        border-radius: 8px;
        margin: auto;
        display: block;
    }
    </style>
""", unsafe_allow_html=True)

# Form to collect user inputs
with st.form("prediction_form"):
    col1, col2 = st.columns(2)

    with col1:
        mh4 = st.selectbox("Medical History 4", options=[1, 2])
        bmi = st.number_input("BMI", min_value=15.0, max_value=55.0, value=24.0)
        pi4 = st.slider("Product Info 4", 0.0, 1.0, 0.3)
        ii6 = st.selectbox("Insured Info 6", options=[1, 2])
        age = st.number_input("Age (Years)", 18, 90, 45)
        fh1 = st.slider("Family History 1", 1, 3, 2)

    with col2:
        ih9 = st.selectbox("Insurance History 9", options=[1, 2, 3])
        weight = st.number_input("Weight (kg)", 40.0, 120.0, 70.0)
        pi2 = st.slider("Product Info 2", 0, 18, 12)
        mh1 = st.slider("Medical History 1", 0, 17, 5)
        ih3 = st.selectbox("Insurance History 3", options=[1, 2, 3])
        fh4_real = st.number_input("Family History 4", 0.2, 0.7, 0.43)

    submitted = st.form_submit_button("Predict")

# ----------------------------
# Prediction logic
if submitted:
    # Normalize BMI, Age, Weight using same logic as Power BI
    bmi_norm = (bmi - 15) / 40
    age_norm = (age - 18) / 72
    weight_norm = (weight - 40) / 80

    # Normalize remaining fields using min-max dictionary
    input_data = {
        'Medical_History_4': normalize_input(mh4, *min_max_dict['Medical_History_4']),
        'BMI': bmi_norm,
        'Product_Info_4': normalize_input(pi4, *min_max_dict['Product_Info_4']),
        'InsuredInfo_6': normalize_input(ii6, *min_max_dict['InsuredInfo_6']),
        'Ins_Age': age_norm,
        'Family_Hist_1': normalize_input(fh1, *min_max_dict['Family_Hist_1']),
        'Insurance_History_9': normalize_input(ih9, *min_max_dict['Insurance_History_9']),
        'Wt': weight_norm,
        'Product_Info_2': normalize_input(pi2, *min_max_dict['Product_Info_2']),
        'Medical_History_1': normalize_input(mh1, *min_max_dict['Medical_History_1']),
        'Insurance_History_3': normalize_input(ih3, *min_max_dict['Insurance_History_3']),
        'Family_Hist_4': normalize_input(fh4_real, *min_max_dict['Family_Hist_4']),
    }

    input_array = np.array([list(input_data.values())])
    prediction = model.predict(input_array)[0]
    final_response = prediction + 1  # Convert 0–7 → 1–8

    # Optional risk bucket
    if final_response in [1, 2, 3]:
        risk = "🟢 Low Risk"
    elif final_response in [4, 5, 6]:
        risk = "🟡 Moderate Risk"
    else:
        risk = "🔴 High Risk"

    st.success(f"✅ Predicted Response: **{final_response}**")
    st.info(f"📊 Risk Category: {risk}")
