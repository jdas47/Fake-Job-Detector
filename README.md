# Fake Job Posting Detector

[🌐 View the Live App](https://fake-job-detector-maabqiij4gtvfnhtfl92et.streamlit.app/)

🚀 A machine learning project to detect fake job postings using Natural Language Processing and classification models.

---

## 📝 Project Description

Fake job postings can scam job seekers into sharing personal data or even paying fake fees. This project analyzes job descriptions and metadata to **predict whether a job posting is real or fake**.

This solution uses:

✅ Text analysis with **TF-IDF vectorization**  
✅ Handling class imbalance using **SMOTE**  
✅ Model training with **Random Forest Classifier**  
✅ Threshold tuning for better recall on fake jobs  
✅ Deployment as a **Streamlit web app**

---

## 📂 Project Structure

├── app.py # Streamlit web app
├── fake_job_detector_model.pkl # Trained model
├── tfidf_vectorizer.pkl # Trained TF-IDF vectorizer
├── requirements.txt # Dependencies
├── README.md # Project documentation


---

## 📊 Dataset

- Source: [Kaggle - Fake Job Postings Dataset](https://www.kaggle.com/datasets/hassanlegends/fake-job-postings)
- ~17,000 real jobs
- ~870 fake jobs
- Text fields: title, description, requirements, benefits, etc.

---

## 💡 Features Used

- **TF-IDF** features extracted from job descriptions
- Metadata flags:
    - `benefits_missing` → whether benefits field was missing
    - `salary_range_missing` → whether salary range was missing

---

## ⚙️ How It Works

1. Loads your job description
2. Transforms it into TF-IDF vectors
3. Adds any missing-field flags
4. Predicts probability of the job being fake
5. Returns:
    - Class label → Fake or Genuine
    - Probability score
---

## 🖥️ Run Locally

### 1. Clone the Repo

```bash
git clone https://github.com/jdas47/Fake-Job-Detector.git
cd fake-job-detector
```

### 2. Install Dependencies
```
pip install -r requirements.txt
```

### 3.  Ensure You Have These Files
```
fake_job_detector_model.pkl
tfidf_vectorizer.pkl
```

### 4.  Run the App
```
streamlit run app.py
```


