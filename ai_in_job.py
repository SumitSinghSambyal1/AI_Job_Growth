import streamlit as st
import pandas as pd
import joblib

# Page ka title aur layout set karein
st.set_page_config(page_title="AI Job Growth Predictor", layout="wide")

# Custom CSS styling
st.markdown("""
<style>
    .main {
        background-color: #000;
    }
    .stApp {
        background-color: #000;
    }
    .stButton > button {
        background-color: #4CAF50 !important;
        color: white !important;
        border-radius: 12px !important;
        padding: 10px 24px !important;
        border: none !important;
        transition: background-color 0.3s ease;
    }

    .stButton > button:hover {
        background-color: #45a049 !important;
    }

    .stButton > button:active, .stButton > button:focus {
        background-color: #388e3c !important;
        color: white !important;
        box-shadow: none !important;
        outline: none !important;
    }

    .stSelectbox, .stTextInput {
        border-radius: 8px;
    }
    h1, h2, h3 {
        color: #1E3A8A;
    }
</style>
""", unsafe_allow_html=True)


# Model aur encoders ko load karein
try:
    model = joblib.load('random_forest_model.pkl')
    label_encoders = joblib.load('label_encoders.pkl')
    le_growth = joblib.load('le_growth.pkl')
except FileNotFoundError:
    st.error("Model ya encoder files nahi mili. Pehle model training script chalaayein.")
    st.stop()


# App ka title
st.title("AI Job Growth Predictor")
st.write("We can predict whether there will be job growth or decline due to AI in different sectors")

# User se input lene ke liye columns banayein
col1, col2, col3 = st.columns(3)

# Har feature ke liye unique values dataset se lein (ya aap manually bhi daal sakte hain)
# Yeh values aapki original CSV file se aani chahiye
job_titles = ['Software Engineer', 'Data Scientist', 'AI Researcher', 'Sales Manager', 'Marketing Specialist', 'HR Manager', 'Product Manager', 'UX Designer', 'Operations Manager', 'Cybersecurity Analyst']
industries = ['Technology', 'Finance', 'Healthcare', 'Retail', 'Education', 'Manufacturing', 'Entertainment', 'Energy', 'Transportation', 'Telecommunications']
company_sizes = ['Small', 'Medium', 'Large']
locations = ['New York', 'San Francisco', 'London', 'Paris', 'Berlin', 'Tokyo', 'Singapore', 'Dubai', 'Sydney', 'Toronto']
ai_adoption_levels = ['Low', 'Medium', 'High']
automation_risks = ['Low', 'Medium', 'High']
required_skills = ['Python', 'Machine Learning', 'Data Analysis', 'Communication', 'Project Management', 'Sales', 'Marketing', 'Cybersecurity', 'UX/UI Design', 'JavaScript']
remote_friendly_options = ['Yes', 'No']


with col1:
    job_title = st.selectbox("Job Title", options=job_titles)
    industry = st.selectbox("Industry", options=industries)
    company_size = st.selectbox("Company Size", options=company_sizes)

with col2:
    location = st.selectbox("Location", options=locations)
    ai_adoption = st.selectbox("AI Adoption Level", options=ai_adoption_levels)
    automation_risk = st.selectbox("Automation Risk", options=automation_risks)

with col3:
    skills = st.selectbox("Required Skills", options=required_skills)
    remote = st.selectbox("Remote Friendly", options=remote_friendly_options)


# Predict button
if st.button("Predict Job Growth"):
    # User input se ek DataFrame banayein
    input_data = {
        'Job_Title': [job_title],
        'Industry': [industry],
        'Company_Size': [company_size],
        'Location': [location],
        'AI_Adoption_Level': [ai_adoption],
        'Automation_Risk': [automation_risk],
        'Required_Skills': [skills],
        'Remote_Friendly': [remote]
    }
    input_df = pd.DataFrame(input_data)

    st.write("---")
    st.subheader("Input:")
    st.dataframe(input_df)

    input_df_encoded = input_df.copy()
    for col in label_encoders:
        input_df_encoded[col] = input_df_encoded[col].apply(lambda x: label_encoders[col].transform([x])[0] if x in label_encoders[col].classes_ else -1)


    prediction_encoded = model.predict(input_df_encoded)
    prediction = le_growth.inverse_transform(prediction_encoded)

    st.subheader("Prediction Result:")
    if prediction[0] == 'Growth':
        st.success(f"**Growth Potential: {prediction[0]}**")
        #st.balloons()
    elif prediction[0] == 'Stable':
        st.info(f"**Growth Potential: {prediction[0]}**")
    else:
        st.warning(f"**Growth Potential: {prediction[0]}**")

