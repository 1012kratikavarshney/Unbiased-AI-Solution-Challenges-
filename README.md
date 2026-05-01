# ⚖️ FairSight AI
### Measure • Flag • Fix AI Bias — Powered by Google Gemini

> **The only no-code AI bias detection platform built for non-technical decision makers.**
> Upload a dataset. Detect bias in seconds. Fix it with one click. Download a professional audit report.


[![Built with Streamlit](https://img.shields.io/badge/Built%20with-Streamlit-FF4B4B?style=flat&logo=streamlit)](https://streamlit.io)
[![Google Gemini](https://img.shields.io/badge/Powered%20by-Google%20Gemini-4285F4?style=flat&logo=google)](https://ai.google.dev)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10+-3776AB?style=flat&logo=python)](https://python.org)

---

## 🚨 The Problem

AI systems now decide who gets hired, who gets a loan, and who receives medical care. These systems silently learn from biased historical data and repeat discrimination at massive scale — without anyone checking.

- **Amazon's hiring AI** rejected women for 4 years — undetected
- **COMPAS algorithm** labeled Black defendants as high-risk at 2× the rate
- **138 million patients** were affected by a biased US hospital AI system
- **50M+ loan applications** in India are decided by AI annually — no fairness audit exists

**No accessible tool exists for the people who make these decisions.** IBM AIF360, Microsoft Fairlearn, and Google What-If Tool all require Python expertise. FairSight AI fixes that.

---

## ✅ Our Solution

FairSight AI is a web platform where any organization can:

1. **Upload** any CSV dataset (hiring, loan, medical, education, government)
2. **Detect** bias automatically using 3 industry-standard metrics
3. **Understand** bias through Google Gemini plain-language explanation
4. **Fix** bias with 1 click — no coding required
5. **Download** a professional AI-generated audit report

**Time to complete:** Under 5 minutes. **Cost:** Free.

---

## 🖥️ Live Demo

> 🔗 **[Try FairSight AI →](https://fairsight-ai.streamlit.app)**
>
> No account needed. Use built-in sample datasets. Add your free Gemini API key from [ai.google.dev](https://ai.google.dev).

---

## 📸 Screenshots

| Detect Bias | AI Explanation | Fix & Compare |
|-------------|---------------|---------------|
| ![detect](screenshots/Detect%20Bias.png) | ![explain](screenshots/AI%20Explanation.png) | ![fix](screenshots/Fix%20%26%20Compare.png) |


---

## ⚙️ How It Works

```
User uploads CSV
      ↓
RandomForest model trained automatically
      ↓
Fairlearn calculates: DPD + EOD + Accuracy Gap
      ↓
Fairness Score (0–100) + Red/Yellow/Green Flag
      ↓
Google Gemini explains in plain language
      ↓
1-click debiasing → Before vs After comparison
      ↓
Gemini generates professional audit report → Download
```

---

## 🗂️ Project Structure

```
fairsight-ai/
├── app.py                  ← Main Streamlit application
├── requirements.txt        ← All dependencies
├── .env.example            ← API key template
├── core/
│   ├── bias_detector.py    ← Bias measurement (DPD, EOD, Accuracy Gap)
│   ├── bias_flagger.py     ← Severity flagging + deploy verdict
│   └── debiaser.py         ← Re-weighting + Fairness Constraint fix
├── ai/
│   └── gemini_service.py   ← Google Gemini: explain + chat + report
└── sample_data/
    ├── hiring_sample.csv   ← Gender bias demo (1000 rows)
    ├── loan_sample.csv     ← Age bias demo (1000 rows)
    └── medical_sample.csv  ← Region bias demo (1000 rows)
```

---

## 🚀 Quick Start

**1. Clone the repository**
```bash
git clone https://github.com/your-username/fairsight-ai.git
cd fairsight-ai
```

**2. Install dependencies**
```bash
pip install -r requirements.txt
```

**3. Add your Gemini API key**
```bash
cp .env.example .env
# Add your key: GEMINI_API_KEY=your_key_here
# Get free key at: https://ai.google.dev
```

**4. Run the app**
```bash
streamlit run app.py
```

**5. Open browser** → `http://localhost:8501`

---

## 🛠️ Tech Stack

| Layer | Technology |
|-------|-----------|
| **UI / Frontend** | Streamlit, Plotly |
| **Google AI** | Google Gemini 1.5 Flash API |
| **Machine Learning** | Scikit-learn (RandomForest, LabelEncoder) |
| **Bias Detection** | Fairlearn (DPD, EOD, MetricFrame) |
| **Bias Fixing** | Fairlearn (ExponentiatedGradient + DemographicParity) |
| **Data Processing** | Pandas, NumPy |
| **Deployment** | Streamlit Cloud / Google Cloud Run |

---

## 📊 Bias Metrics Explained

| Metric | What It Measures | Red Flag If |
|--------|-----------------|-------------|
| **Demographic Parity Difference (DPD)** | Selection rate gap between groups | > 0.1 |
| **Equalized Odds Difference (EOD)** | Accuracy gap between groups | > 0.1 |
| **Disparate Impact Ratio** | Ratio of selection rates (legal standard) | < 0.8 |
| **Fairness Score** | Overall 0–100 composite score | < 65 |

---

## 🆚 Why FairSight AI is Different

| Feature | IBM AIF360 | Fairlearn | What-If Tool | **FairSight AI** |
|---------|-----------|-----------|--------------|-----------------|
| No-Code Interface | ✗ | ✗ | ✗ | **✓** |
| CSV Upload & Auto-Analyze | ✗ | ✗ | ✗ | **✓** |
| AI Plain-Language Explanation | ✗ | ✗ | ✗ | **✓** |
| 1-Click Bias Fix | ✗ | ✗ | ✗ | **✓** |
| AI Audit Report | ✗ | ✗ | ✗ | **✓** |
| Deploy / Don't Deploy Verdict | ✗ | ✗ | ✗ | **✓** |
| Any Domain Support | ✓ | ✓ | ✗ | **✓** |
| Free to Use | ✓ | ✓ | ✓ | **✓** |

---

## 📋 Sample Datasets Included

| Dataset | Domain | Bias Type | Rows |
|---------|--------|-----------|------|
| `hiring_sample.csv` | HR / Recruitment | Gender bias (+18% male advantage) | 1,000 |
| `loan_sample.csv` | Banking / Finance | Age bias (-20% for 18–30 group) | 1,000 |
| `medical_sample.csv` | Healthcare | Region bias (-25% for rural areas) | 1,000 |

---

## 🔮 Future Roadmap

- [ ] Real-time bias monitoring with email/SMS alerts
- [ ] SHAP value integration for feature-level bias explanation
- [ ] EU AI Act 2024 + India DPDP Act 2023 compliance checker
- [ ] Multi-language support (Hindi, Tamil, Bengali)
- [ ] FairSight REST API for pipeline integration
- [ ] Industry benchmarking — compare your score vs sector average
- [ ] Intersectional bias detection (gender + age + region combined)

---

## 🏆 Hackathon Submission

This project was built for the **Google Solution Challenge 2026**.

- **Problem Statement:** [Unbiased AI Decision] — Ensuring Fairness and Detecting Bias in Automated Decisions
- **Google Technology Used:** Google Gemini 1.5 Flash API
- **Category:** AI / Social Impact

---

## 🙏 Acknowledgements

- [Google Gemini](https://ai.google.dev) — AI explanation and report generation
- [Microsoft Fairlearn](https://fairlearn.org) — Bias metrics and debiasing algorithms
- [Streamlit](https://streamlit.io) — Web application framework
- [ProPublica COMPAS Analysis](https://github.com/propublica/compas-analysis) — Real-world bias research

---

<div align="center">

**⚖️ FairSight AI — Because fairness should be guaranteed, not optional.**

[🔗 Live Demo](https://fairsight-ai.streamlit.app/) • [📹 Demo Video](https://drive.google.com/file/d/1xjXcKF-1B_nXrCvq6t5smu5HlYXlfe_S/view?usp=sharing)

</div>

