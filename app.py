import streamlit as st
import pandas as pd
import numpy as np
import joblib

# --- Page Configuration ---
st.set_page_config(
    page_title="Resume Quality Predictor",
    page_icon="📄",
    layout="centered"
)

# --- Load Models & Assets ---
knn_model = joblib.load("knn_model.pkl")
logistic_model = joblib.load("logistic_model.pkl")
tree_model = joblib.load("tree_model.pkl")
scaler_knn = joblib.load("scaler_knn.pkl")
scaler_logistic = joblib.load("scaler_logistic.pkl")
tree_columns = joblib.load("tree_features.pkl")

# --- Title ---
st.markdown("""
    <h1 style="text-align: center; color: #2c3e50;">📄 Resume Quality Predictor</h1>
    <p style="text-align: center; font-size: 18px;">
        Choose from 3 powerful ML models to predict if a resume will receive a callback!
    </p>
""", unsafe_allow_html=True)

# --- Model Selection ---
st.markdown("### 🧠 Choose a Prediction Model")
model_choice = st.radio(
    "Select Model to Use:",
    ["🔷 KNN", "📈 Logistic Regression", "🌳 Decision Tree"],
    horizontal=True,
    label_visibility="collapsed"
)

st.divider()

# --- Input Sections ---
st.markdown("## ✍️ Candidate Information")

# KNN MODEL
if "KNN" in model_choice:
    with st.form("knn_form"):
        col1, col2 = st.columns(2)
        with col1:
            has_email = st.selectbox("📧 Has Email Address?", ["No", "Yes"])
            volunteer = st.selectbox("🤝 Volunteered?", ["No", "Yes"])
        with col2:
            military = st.selectbox("🎖️ Military Experience?", ["No", "Yes"])
            worked_during_school = st.selectbox("📚 Worked During School?", ["No", "Yes"])

        years_experience = st.slider("💼 Years of Experience", 0.0, 40.0, step=0.5)

        if st.form_submit_button("🔍 Predict"):
            input_df = pd.DataFrame([{
                "has_email_address": 1 if has_email == "Yes" else 0,
                "volunteer": 1 if volunteer == "Yes" else 0,
                "worked_during_school": 1 if worked_during_school == "Yes" else 0,
                "military": 1 if military == "Yes" else 0,
                "years_experience": years_experience
            }])

            scaled = scaler_knn.transform(input_df)
            pred = knn_model.predict(scaled)[0]
            if pred == 1:
                st.success("✅ **Prediction:** High Quality Resume")
            else:
                st.error("❌ **Prediction:** Low Quality Resume")

# LOGISTIC REGRESSION
elif "Logistic" in model_choice:
    with st.form("log_form"):
        years_experience = st.slider("💼 Years of Experience", 0.0, 40.0, step=0.5)
        computer_skills = st.radio("💻 Has Computer Skills?", ["No", "Yes"])
        college_degree = st.radio("🎓 Has College Degree?", ["No", "Yes"])
        employment_holes = st.radio("🕳️ Employment Gaps?", ["No", "Yes"])
        worked_during_school = st.radio("📚 Worked During School?", ["No", "Yes"])

        if st.form_submit_button("🔍 Predict"):
            input_df = pd.DataFrame([{
                "years_experience": years_experience,
                "computer_skills": 1 if computer_skills == "Yes" else 0,
                "college_degree": 1 if college_degree == "Yes" else 0,
                "employment_holes": 1 if employment_holes == "Yes" else 0,
                "worked_during_school": 1 if worked_during_school == "Yes" else 0
            }])

            scaled = scaler_logistic.transform(input_df)
            pred = logistic_model.predict(scaled)[0]
            if pred == 1:
                st.success("📞 **Prediction:** Callback Likely")
            else:
                st.warning("📭 **Prediction:** Callback Unlikely")

# DECISION TREE
elif "Tree" in model_choice:
    with st.form("tree_form"):
        with st.expander("🏢 Job Information", expanded=True):
            job_city = st.selectbox("City", ['Boston', 'Chicago'])
            job_industry = st.selectbox("Industry", [
                'manufacturing', 'wholesale_and_retail_trade', 'business_and_personal_service',
                'finance_insurance_real_estate', 'transportation_communication', 'other_service'
            ])
            job_type = st.selectbox("Job Type", ['supervisor', 'secretary', 'sales_rep', 'retail_sales', 'manager', 'clerical'])
            job_ownership = st.selectbox("Ownership", ['nonprofit', 'public', 'private', 'unknown'])
            job_req_school = st.selectbox("Education Level", ['high_school_grad', 'some_college', 'college', 'non_listed'])

        with st.expander("👤 Candidate Attributes", expanded=True):
            col1, col2 = st.columns(2)
            with col1:
                job_req_any = st.radio("Any Job Requirements?", [0, 1])
                job_req_education = st.radio("Education Requirement?", [0, 1])
                job_req_min_experience = st.number_input("Min Experience Required", 0.0)
                job_req_computer = st.radio("Computer Skills Required?", [0, 1])
                job_req_organization = st.radio("Org Skills Required?", [0, 1])
            with col2:
                race = st.selectbox("Race", ['white', 'black'])
                gender = st.selectbox("Gender", ['m', 'f'])
                years_college = st.slider("Years in College", 0, 4)
                college_degree = st.radio("College Degree?", [0, 1])

        with st.expander("📝 Resume Traits", expanded=True):
            honors = st.radio("Has Honors?", [0, 1])
            worked_during_school = st.radio("Worked During School?", [0, 1])
            years_experience = st.number_input("Total Work Experience", 0.0)
            computer_skills = st.radio("Computer Skills?", [0, 1])
            special_skills = st.radio("Special Skills?", [0, 1])
            military = st.radio("Military Background?", [0, 1])
            employment_holes = st.radio("Employment Gaps?", [0, 1])

        if st.form_submit_button("🔍 Predict"):
            test_data = {
                'job_city': job_city,
                'job_industry': job_industry,
                'job_type': job_type,
                'job_ownership': job_ownership,
                'job_req_school': job_req_school,
                'job_req_any': job_req_any,
                'job_req_education': job_req_education,
                'job_req_min_experience': job_req_min_experience,
                'job_req_computer': job_req_computer,
                'job_req_organization': job_req_organization,
                'race': race,
                'gender': gender,
                'years_college': years_college,
                'college_degree': college_degree,
                'honors': honors,
                'worked_during_school': worked_during_school,
                'years_experience': years_experience,
                'computer_skills': computer_skills,
                'special_skills': special_skills,
                'military': military,
                'employment_holes': employment_holes
            }

            df = pd.DataFrame([test_data])
            encoded = pd.get_dummies(df).reindex(columns=tree_columns, fill_value=0)
            pred = tree_model.predict(encoded)[0]
            if pred == 1:
                st.success("📥 **Prediction:** Callback Will Be Received ✅")
            else:
                st.error("📤 **Prediction:** Callback Not Expected ❌")
