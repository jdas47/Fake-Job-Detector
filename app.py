
import streamlit as st
import joblib
import pandas as pd

st.title("Fake Job Posting Detector")

# Load model and vectorizer
model = joblib.load("fake_job_detector_model.pkl")
vectorizer = joblib.load("tfidf_vectorizer.pkl")

description = st.text_area("Job Description", "")
benefits_missing = st.checkbox("Benefits Missing?", value=True)
salary_missing = st.checkbox("Salary Range Missing?", value=True)

threshold = st.slider("Prediction Threshold", 0.0, 1.0, 0.3)

def predict(description, benefits_missing, salary_missing):
    if not description:
        return None
    desc_vec = vectorizer.transform([description])
    desc_df = pd.DataFrame(desc_vec.toarray(), columns=vectorizer.get_feature_names_out())
    extra = pd.DataFrame([[int(benefits_missing), int(salary_missing)]], columns=["benefits_missing", "salary_range_missing"])
    features = pd.concat([desc_df, extra], axis=1)
    prob = model.predict_proba(features)[0][1]
    return prob

if st.button("Predict"):
    prob = predict(description, benefits_missing, salary_missing)
    prediction = "Fake Job" if prob >= threshold else "Genuine Job"
    st.write(f"**Result**: {prediction}")
    st.write(f"**Probability of being fake**: {prob:.2f}")
