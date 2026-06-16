# 🧬 Cancer Survival Predictor

An AI-powered web application that predicts cancer patient survival outcomes using a Decision Tree Classifier, built with data from Indian patient demographics.

🔗 **Live App:** [https://india-cancer-patient-predictor-6.onrender.com](https://india-cancer-patient-predictor-6.onrender.com)

---

## 📋 Overview

This tool allows healthcare professionals and researchers to input patient details and receive an instant survival prediction. It is designed for research and educational purposes only and is not intended for clinical diagnosis.

---

## ✨ Features

- Predicts cancer survival outcome based on patient demographics and clinical data
- Covers **15 cancer types** common in the Indian population
- Supports **14 Indian states** and **11 major cities**
- Considers **7 treatment types**
- Powered by a Decision Tree Classifier model
- Simple, intuitive web interface — no setup required

---

## 🧾 Input Parameters

| Category           | Field            | Options / Type                                                                                                                         |
|--------------------|------------------|----------------------------------------------------------------------------------------------------------------------------------------|
| **Patient Info**   | Age              | Numeric                                                                                                                                |
|                    | Gender           | Male, Female, Other                                                                                                                    |
|                    | State            | Andhra Pradesh, Bihar, Delhi, Gujarat, Karnataka, Kerala, Madhya Pradesh, Maharashtra, Punjab, Rajasthan, Tamil Nadu, Telangana, Uttar Pradesh, West Bengal |
|                    | City             | Ahmedabad, Bangalore, Chennai, Delhi, Hyderabad, Jaipur, Kolkata, Lucknow, Mumbai, Pune, Surat                                        |
| **Clinical Info**  | Cancer Type      | Bladder, Brain, Breast, Cervical, Colon, Kidney, Leukemia, Liver, Lung, Ovarian, Pancreatic, Prostate, Skin, Stomach, Thyroid         |
|                    | Stage            | Stage I, Stage II, Stage III, Stage IV                                                                                                 |
|                    | Treatment Type   | Chemotherapy, Combined, Hormone Therapy, Immunotherapy, Radiation, Surgery, Targeted Therapy                                           |
|                    | Survival Months  | Numeric                                                                                                                                |

---

## 🤖 Model

- **Algorithm:** Decision Tree Classifier
- **Framework:** scikit-learn
- **Task:** Binary/multi-class survival prediction
- **Training Data:** Indian cancer patient dataset

---

## 🚀 Deployment

The app is deployed on **Render** and is publicly accessible at:

```
https://india-cancer-patient-predictor-6.onrender.com
```

> **Note:** Since Render's free tier spins down after inactivity, the first load may take 30–60 seconds.

---

## 🛠️ Tech Stack

| Layer       | Technology                  |
|-------------|-----------------------------|
| ML Model    | scikit-learn (Decision Tree) |
| Backend     | Python (FastAPI / Flask)    |
| Frontend    | HTML, CSS, JavaScript       |
| Hosting     | Render                      |

---

## ⚠️ Disclaimer

This application is built **for research and educational purposes only**. Predictions made by this tool should **not** be used as a substitute for professional medical advice, diagnosis, or treatment. Always consult a qualified healthcare provider for medical decisions.

---

## 📬 Contact

For questions, feedback, or contributions, please open an issue or reach out via the repository.
