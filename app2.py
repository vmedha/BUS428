import streamlit as st
import pickle
import pandas as pd

# Page settings
st.set_page_config(page_title="Data Science Salary Estimator", page_icon="💻", layout="wide")

# Load the trained model
@st.cache_resource
def load_model():
    with open("salary2025_model.pkl", "rb") as f:
        return pickle.load(f)

model = load_model()

# Title
st.title("🎯 Data Science Salary Estimator")
st.write("Fill out your details below to get an estimated salary prediction based on your profile.")

# Sidebar for user input
with st.sidebar:
    st.header("Profile Information")
    
    education_options = {'High School': 0, 'Bachelor\'s Degree': 1, 'Master\'s Degree': 2, 'PhD': 3}
    education = st.selectbox("Select your highest education level", list(education_options.keys()))
    years_coding = st.slider("Years of Programming Experience", 0, 40, 3)
    country = st.selectbox("Select your country", ["United States", "United Kingdom", "Germany", "Other"])

    st.subheader("Programming Languages Known:")
    codes_java = st.toggle("Java")
    codes_python = st.toggle("Python")
    codes_sql = st.toggle("SQL")
    codes_go = st.toggle("Go")

# Processing inputs
education_num = education_options[education]
skills_known = sum([codes_java, codes_python, codes_sql, codes_go])

# Prepare feature input
features = {
    "Education": education_num,
    "YearsCoding": years_coding,
    "Java": int(codes_java),
    "Python": int(codes_python),
    "SQL": int(codes_sql),
    "Go": int(codes_go),
    "SkillCount": skills_known,   
    "Country_Germany": 0,
    "Country_United Kingdom": 0,
    "Country_United States": 0,
}

# Set country dummy
if country == "United States":
    features["Country_United States"] = 1
elif country == "United Kingdom":
    features["Country_United Kingdom"] = 1
elif country == "Germany":
    features["Country_Germany"] = 1

# Convert features to DataFrame
input_df = pd.DataFrame([features])

# Salary prediction
st.subheader("📊 Salary Prediction Result")

if st.button("Estimate My Salary 💸"):
    salary_pred = model.predict(input_df.values)[0]
    salary_low = salary_pred * 0.85
    salary_high = salary_pred * 1.15

    st.metric(label="Predicted Salary (USD)", value=f"${salary_pred:,.0f}")
    st.info(f"Estimated range: ${salary_low:,.0f} - ${salary_high:,.0f}")

st.markdown("---")
st.caption("Powered by Streamlit • Developed by [Medha V]")

