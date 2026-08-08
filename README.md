# 🎮 Cookie Cats Retention Intelligence & Early Churn Prediction Engine

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://cookie-cats-retention-intelligence.streamlit.app/)
![Python 3.12](https://img.shields.io/badge/Python-3.12-3776AB?style=flat&logo=python&logoColor=white)
![Scikit-Learn 1.6.1](https://img.shields.io/badge/Scikit--Learn-1.6.1-F7931E?style=flat&logo=scikit-learn&logoColor=white)
![LightGBM](https://img.shields.io/badge/Model-LightGBM-GREEN?style=flat)
![Deployment](https://img.shields.io/badge/Deployment-Streamlit%20Cloud-FF4B4B?style=flat&logo=streamlit&logoColor=white)

An end-to-end Data Science, A/B Testing Analysis, and Machine Learning deployment project evaluating player progression mechanics and early retention predictors in the mobile puzzle game **Cookie Cats**. 

🔗 **Live Web Application:** [Cookie Cats Retention Intelligence Hub](https://cookie-cats-retention-intelligence.streamlit.app/)

---

## 📌 Executive Summary

When designing free-to-play mobile games, progression gates (forcing players to wait or make an in-app purchase before continuing) serve as crucial engagement boundaries. This project analyzes an A/B test conducted on **90,189 players** to determine whether moving the first gate from **Level 30** (*Control*) to **Level 40** (*Test*) impacts long-term player retention.

Additionally, we engineered a supervised Machine Learning pipeline (utilizing LightGBM, XGBoost, and Decision Trees) to predict **7-Day Retention** from early player behavior, achieving automated early-churn risk detection.

---

## 🛠️ Tech Stack & Ecosystem

* **Core Language:** Python 3.12
* **Machine Learning & Modeling:** Scikit-Learn 1.6.1, LightGBM, XGBoost, Decision Trees
* **Data Manipulation & Analysis:** Pandas, NumPy, SciPy
* **Data Visualization:** Plotly Express, Plotly Graph Objects, Seaborn, Matplotlib
* **Model Serialization & Production:** Joblib
* **Deployment & Web Framework:** Streamlit, Streamlit Cloud

---

## 📊 Key A/B Testing Insights & Findings

1. **Gate Placement Directly Impacts Retention:**
   * **Gate 30 (Control):** 1-Day Retention: **44.82%** | 7-Day Retention: **19.02%**
   * **Gate 40 (Test):** 1-Day Retention: **44.23%** | 7-Day Retention: **18.20%**
   * **Conclusion:** Delaying the gate to Level 40 caused a **0.82% absolute drop** in long-term 7-day retention.

2. **Statistical Confidence (Bootstrap Analysis):**
   * Repeated bootstrap resampling (10,000 iterations) yields a **99.90% probability** that Gate 30 leads to superior 7-day retention compared to Gate 40.

3. **Behavioral Insight (Hedonic Adaptation):**
   * Enforcing an earlier gate at Level 30 forces players into short, scheduled breaks before game burnout sets in. Pushing the gate back to Level 40 causes continuous play, increasing player fatigue and sudden churn.

4. **Early Onboarding Drop-Off:**
   * Over **4.4% of users (3,994 players)** opened the game but played 0 rounds.
   * Almost **20% of users** churned before reaching round 5, proving that initial onboarding friction is the single largest leak in the player funnel.

---

## 💡 Key Behavioral Q&A Derived From Findings

* **Q1: How does player engagement differ between retained and churned cohorts?**
  * *Answer:* Retained 7-day players log a median of **108 rounds**, whereas churned users drop off after a median of just **11 rounds**. Early engagement frequency in Days 1–3 dictates long-term player lifetime value.
* **Q2: Should product managers delay progression gates to let players play longer?**
  * *Answer:* No. Bootstrap simulation confirms with **99.9% certainty** that delaying gates decreases long-term retention due to player exhaustion.
* **Q3: What segment of installs experiences immediate drop-off?**
  * *Answer:* ~4.4% register zero game rounds, pointing toward onboarding or technical loading bottlenecks prior to actual level interaction.

---

## 🚀 Machine Learning & Pipeline Architecture

* **Problem Formulation:** Binary Classification / Churn Risk Prediction (Target: `retention_7`).
* **Feature Engineering:** Log-transformed engagement metrics (`log_gamerounds`), Day-1 interaction features (`ret1_x_rounds`), and zero-play flags.
* **Model Selection & Tuning:** Compared Decision Trees, XGBoost, and LightGBM. LightGBM optimized with custom threshold tuning (`best_threshold.pkl`) yielded top ROC-AUC performance.
* **Production Deployment:** Serialized model artifacts via `joblib` integrated into a custom glassmorphism Streamlit UI.

---

## 📁 Repository Structure

```text
├── app.py                      # Production Streamlit Web Interface Code
├── lightgbm_model.pkl          # Serialized LightGBM Model Pipeline
├── best_threshold.pkl         # Optimized Decision Threshold Matrix
├── requirements.txt            # Environment Dependencies (Scikit-Learn 1.6.1)
├── cookie_cats.csv             # Raw A/B Testing Dataset
└── README.md                   # Project Documentation
```

## 📞 Contact Information:-
* **Email:-**[englandengland271@gmail.com]
* **Linkedin:-**[https://www.linkedin.com/in/mohammed-nafay-ali-16519138a?utm_source=share_via&utm_content=profile&utm_medium=member_android]
* **GitHub:-**[https://github.com/M-Nafay-Ali]
  
