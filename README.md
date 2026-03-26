# 🏥 Healthcare No-Show Prediction

> Predicting patient no-shows after reminder interventions using machine learning — and translating model outputs into a risk-tiered clinical intervention framework.

**Dataset:** 110,527 real patient appointment records from Brazil (Kaggle)  
**Best Model:** XGBoost — Validation AUC **0.748**  
**Tools:** R · caret · xgboost · randomForest · ggplot2 · pROC  

---

## 📌 Project Overview

Clinics and hospitals send SMS and call reminders to reduce no-shows — yet many patients still fail to attend. This project goes beyond model accuracy to answer a more actionable question:

> **If we already sent a reminder, which patients are still likely to no-show — and what should we do differently for them?**

Four models were built, compared, and evaluated. The best-performing model (XGBoost, AUC 0.748) was used to power a **risk-tiered intervention framework** with concrete recommendations for clinic operations and CRM teams.

---

## 📊 Model Comparison

| Model | Validation AUC | vs Baseline |
|---|---|---|
| Logistic Regression | 0.678 | — |
| Logistic Regression + SMOTE | 0.679 | +0.001 |
| Random Forest | 0.715 | +0.037 |
| Random Forest + SMOTE | 0.725 | +0.047 |
| **XGBoost ✅ Selected** | **0.748** | **+0.070** |

> Training AUC (LR): 0.674 vs Validation: 0.678 — near-zero gap confirms no overfitting.

---

## 🔍 Key Findings

### 1. Saturday is the Highest No-Show Day — Not Monday
| Day | No-Show Rate |
|---|---|
| Saturday | **23.1%** ← Highest |
| Friday | 21.2% |
| Monday | 20.6% |
| Thursday | 19.3% ← Lowest |

**Recommendation:** Hold 2–3 buffer slots on Saturdays for same-day waitlist fills.

---

### 2. Teenagers Are the Highest-Risk Age Group
| Age Group | No-Show Rate |
|---|---|
| Teen (13–18) | **26.1%** |
| Young Adult (19–35) | 23.8% |
| Child (0–12) | 20.5% |
| Middle Age (36–60) | 19.1% |
| Senior (61+) | **15.2%** ← Lowest |

---

### 3. SMS Reminders Are Counterproductive in Current Form
Patients who received SMS reminders showed **higher** no-show rates than those who did not — across almost every age group.

| Age Group | No SMS | SMS Sent |
|---|---|---|
| Teen (13–18) | 22.2% | **28.8%** |
| Young Adult (19–35) | 18.5% | **35.5%** |
| Senior (61+) | 13.4% | **19.2%** |

**Implication:** Current SMS content and targeting requires redesign. Generic reminders may create false confirmation rather than driving attendance.

---

### 4. Top Predictors (Logistic Regression Odds Ratios)

| Predictor | Odds Ratio | Direction |
|---|---|---|
| High-Risk Flag (lead >5d & prior no-shows >1) | ~1.65 | 🔴 Increases risk |
| SMS Received | ~1.45 | 🔴 Increases risk |
| Prior No-Show History | ~1.30 | 🔴 Increases risk |
| Scholarship (welfare) | ~1.25 | 🔴 Increases risk |
| Alcoholism | ~1.20 | 🔴 Increases risk |
| Hypertension | ~0.90 | 🟢 Decreases risk |
| Age (per decade) | ~0.79 | 🟢 Decreases risk |

---

## 🎯 Intervention Recommendation Matrix

| Risk Tier | P(No-Show) | Action | Owner |
|---|---|---|---|
| 🔴 HIGH | ≥40% | Personal phone call 48hrs before + flexible slot swap + double-booking protocol | Clinic Coordinator |
| 🟡 MEDIUM | 25–39% | Personalised SMS + 1-tap reschedule link + reminders at 72hrs AND 24hrs | CRM Workflow |
| 🟢 LOW | <25% | Standard automated reminder only | Automated System |

**Optimal decision threshold:** 0.18 (F1-maximised via threshold tuning)

---

## 💰 Business Impact

| Clinic Size | Annual Appointments | Est. Cost of No-Shows | Projected Savings (15% reduction) |
|---|---|---|---|
| Small | 10,000 | $412,000 | ~$61,800/yr |
| Medium | 30,000 | $1,236,000 | ~$185,400/yr |
| Large | 100,000 | $4,120,000 | ~$618,000/yr |

> Assumes: 20.6% baseline no-show rate · $200 cost per no-show (MGMA benchmark) · 15% reduction with tiered intervention

---

## 📁 Repository Structure

```
├── healthcare_noshow_analysis.R   # Full analysis script (Sections 0–11)
├── NoShow_Business_Case_Final.docx  # Business case & intervention framework
├── plots/
│   ├── eda_plots.png              # No-show rates by day, age, lead time, SMS
│   ├── correlation_matrix.png     # Feature correlation heatmap
│   ├── odds_ratios.png            # Logistic regression forest plot
│   ├── roc_logistic_comparison.png  # Full vs Stepwise LR ROC
│   ├── roc_train_vs_validation.png  # Overfitting check
│   └── threshold_tuning.png       # Sensitivity / Specificity / F1 tradeoff
└── README.md
```

---

## 🚀 How to Run

**1. Clone the repository**
```bash
git clone https://github.com/YOUR_USERNAME/healthcare-noshow-prediction
cd healthcare-noshow-prediction
```

**2. Download the dataset**

Download `KaggleV2-May-2016.csv` from [Kaggle](https://www.kaggle.com/datasets/joniarroba/noshowappointments) and place it in the project folder.

**3. Install R dependencies**

```r
install.packages(c("tidyverse", "caret", "pROC", "corrplot",
                   "gridExtra", "scales", "lubridate",
                   "randomForest", "xgboost", "themis"))
```

**4. Run the analysis**

Open `healthcare_noshow_analysis.R` in RStudio and run all sections sequentially (Ctrl+A → Ctrl+Enter). The script will generate all plots and print model results to the console.

---

## 📈 Analysis Pipeline

```
Data Loading → Cleaning & Feature Engineering → EDA → Correlation Analysis
     ↓
Train / Validation / Test Split (60 / 20 / 20)
     ↓
Logistic Regression → Random Forest → XGBoost → SMOTE Variants
     ↓
Threshold Tuning → Risk Stratification → Business Impact Estimation
```

---

## 🛠 Feature Engineering

Beyond the raw dataset variables, the following features were engineered:

- **`lead_time`** — days between scheduling and appointment date
- **`age_group`** — binned age categories (Child / Teen / Young Adult / Middle Age / Senior)
- **`prior_noshow_count`** — cumulative no-show history per patient (calculated in appointment order)
- **`high_risk`** — composite flag: lead time >5 days AND prior no-shows >1
- **`appt_weekday_num`** — numeric day of week (1=Monday, 7=Sunday)

---

## 📋 Dataset

- **Source:** [Medical Appointment No-Shows — Kaggle](https://www.kaggle.com/datasets/joniarroba/noshowappointments)
- **Records:** 110,527 appointments
- **Location:** Brazil, May 2016
- **Target variable:** No-show (Yes/No) → binary 1/0
- **No-show rate:** ~20.6% (weekday average)

---

## 👤 Author

**Aryaa Singh**  
MS Business Analytics & Artificial Intelligence — University of Texas at Dallas  
[LinkedIn](www.linkedin.com/in/aryaasingh) · [GitHub](https://github.com/AryaaSingh07)

---

## 📄 License

MIT License — see [LICENSE](LICENSE) for details.
