# 🎯 SegmentPulse ML - Intelligent Customer Segmentation Studio

[![Python Version](https://img.shields.io/badge/Python-3.9%20%7C%203.10%20%7C%203.11%20%7C%203.12%20%7C%203.13-blue.svg)](https://www.python.org/)
[![Django](https://img.shields.io/badge/Django-4.2+-092e20.svg?logo=django&logoColor=white)](https://www.djangoproject.com/)
[![Scikit-Learn](https://img.shields.io/badge/scikit--learn-1.3+-f7931e.svg?logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![Bootstrap](https://img.shields.io/badge/Bootstrap-5.3-7952b3.svg?logo=bootstrap&logoColor=white)](https://getbootstrap.com/)
[![Chart.js](https://img.shields.io/badge/Chart.js-4.x-ff6384.svg?logo=chartdotjs&logoColor=white)](https://www.chartjs.org/)

An enterprise-grade, full-stack machine learning web application built with **Django** and **Scikit-Learn** for demographic and behavioral customer segmentation. The platform combines supervised ensemble classification (**Random Forest**, **Decision Tree**) with unsupervised geometric clustering (**KMeans**) to automatically categorize customer profiles and generate actionable strategic marketing playbooks.

---

## 📑 Table of Contents

1. [Key Features](#-key-features)
2. [Customer Segments Matrix](#-customer-segments-matrix)
3. [Restructured Project Layout](#-restructured-project-layout)
4. [Quick Start & Execution Guide](#-quick-start--execution-guide)
5. [Feature Guide](#-feature-guide)
6. [Machine Learning Pipeline](#-machine-learning-pipeline)
7. [REST API Documentation](#-rest-api-documentation)
8. [Troubleshooting & FAQ](#-troubleshooting--faq)

---

## 🚀 Key Features

- **⚡ Real-time ML Inference**: Instant multi-class customer segmentation with calibrated confidence scores and latency metrics.
- **✨ Modern Glassmorphic UI**: Beautiful frontend crafted with **Bootstrap 5**, **Google Fonts (Plus Jakarta Sans)**, custom CSS design tokens, smooth micro-interactions, and responsive layouts.
- **🪄 1-Click Persona Presets**: Instantly populate the form with realistic test personas (Corporate Executive, Tech Innovator, Household Builder, Prudent Saver).
- **📊 Interactive Visual Analytics**: Real-time **Chart.js** cohort dashboards showing segment distributions, spending patterns, model usage, and profession breakdown.
- **🗂️ Audit History & Search**: Full paginated archive of customer profiles with segment filtering, profession search, and detailed inspection.
- **📥 CSV Dataset Export**: One-click export of all segmented records into standard CSV format for BI tools.
- **🤖 Dual Engine + Clustering**: Choose between Random Forest (multi-tree ensemble) or Decision Tree (rule-based) with automatic KMeans cluster assignment.
- **🔌 Programmatic REST API**: JSON POST endpoint for integrating customer segmentation directly into mobile apps or CRM pipelines.

---

## 🏷️ Customer Segments Matrix

| Segment | Cohort Name | Key Characteristics | Strategic Focus | Optimal Channels |
| :---: | :--- | :--- | :--- | :--- |
| **A** | **Elite High-Value VIPs** | High spenders, executive professionals, mature career stage, high lifetime value. | VIP loyalty perks, executive events, luxury tier upgrades. | Private Email, Executive Webinars, Concierge Outreach |
| **B** | **Rising Professionals** | Mid-career upwardly mobile professionals, tech-savvy, moderate-to-high spend. | Career growth bundles, digital rewards, flexible payment options. | LinkedIn, Instagram, Mobile Push Notifications |
| **C** | **Household Builders** | Family-focused budget managers, larger household sizes, recurring essentials. | Family packages, bulk discounts, seasonal utility promotions. | Email Newsletters, Community Groups, SMS Reminders |
| **D** | **Prudent Savers** | Price-sensitive younger or retired consumers, cautious trial adopters. | Low-barrier starter tiers, transparent pricing, referral incentives. | Direct SMS, Discount Portals, Search Ads |

---

## 📁 Restructured Project Layout

The repository has been restructured into clean, standardized root-level folders:

```text
Customer Segmentation/
├── customer_segmentation/          # Django application module
│   ├── migrations/                 # Database schema migrations
│   ├── templates/                  # Modern HTML5/Bootstrap 5 templates
│   │   └── customer_segmentation/
│   │       ├── base.html           # Master layout (Google Fonts, Glassmorphism, Navbar, Footer)
│   │       ├── home.html           # Main profiler with 1-click persona presets
│   │       ├── result.html         # High-impact outcome card & strategic playbook
│   │       ├── analytics.html      # Interactive Chart.js cohort analytics
│   │       ├── history.html        # Searchable archive with segment filters & CSV export
│   │       ├── detail.html         # Individual profile deep-dive view
│   │       ├── model_performance.html # Benchmark metrics & feature importance bars
│   │       └── confirm_delete.html # Secure record removal confirmation
│   ├── admin.py                    # Django administration console configuration
│   ├── apps.py                     # App configuration
│   ├── forms.py                    # Clean form widgets and validation rules
│   ├── ml_service.py               # ML training, preprocessing, inference & metrics
│   ├── models.py                   # ORM models (CustomerSegmentation, ModelPerformance)
│   ├── tests.py                    # Automated test suite (9 unit & integration tests)
│   └── urls.py                     # Application route definitions
├── customer_segmentation_project/  # Django project core configuration
│   ├── asgi.py                     # ASGI entry point
│   ├── settings.py                 # Project configuration & settings
│   ├── urls.py                     # Root routing table
│   └── wsgi.py                     # WSGI production server entry point
├── data/                           # Dataset repository
│   └── train.csv                   # Clean training dataset (8,068 rows x 11 columns)
├── models/                         # Serialized trained Scikit-Learn models
│   ├── random_forest.pkl           # Trained Random Forest Classifier
│   ├── decision_tree.pkl           # Trained Decision Tree Classifier
│   ├── kmeans.pkl                  # Trained KMeans clustering model
│   ├── label_encoder.pkl           # Target variable label encoder
│   ├── feature_columns.pkl         # Feature schema ordering
│   ├── metrics.pkl                 # Evaluated model benchmark scores
│   └── feature_importances.pkl     # Gini feature importance rankings
├── notebooks/                      # Exploratory Data Analysis & research
│   └── Customer_Segmentation_EDA.ipynb # Jupyter exploration notebook
├── static/                         # Static web assets
│   ├── css/
│   │   └── style.css               # Design system tokens, segment badges, animations
│   └── js/
│       └── main.js                 # 1-click preset injector & form animations
├── manage.py                       # Root Django management CLI
├── requirements.txt                # Python dependencies
├── .gitignore                      # Git exclusion rules
└── README.md                       # Documentation
```

---

## ⚡ Quick Start & Execution Guide

Follow these simple steps to run the application on your computer:

### 1. Prerequisites
- **Python 3.9+** installed on your system.
- Terminal / PowerShell / Command Prompt.

### 2. Navigate to Project Directory
Open your terminal in the workspace root:
```bash
cd "c:\D drive\ML_Projects\Customer Segmentation"
```

### 3. Create & Activate a Virtual Environment (Recommended)

**On Windows (PowerShell):**
```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

**On Windows (Command Prompt):**
```cmd
python -m venv venv
venv\Scripts\activate.bat
```

**On macOS / Linux:**
```bash
python3 -m venv venv
source venv/bin/activate
```

### 4. Install Dependencies
```bash
pip install -r requirements.txt
```

### 5. Apply Database Migrations
Initialize the SQLite database schema:
```bash
python manage.py migrate
```

### 6. (Optional) Create Admin Superuser
To access the Django administrator dashboard:
```bash
python manage.py createsuperuser
```
Follow the prompt to set your username and password.

### 7. Run Automated Tests (Verification)
Run the built-in test suite to confirm everything is operating correctly:
```bash
python manage.py test
```
*All 9 unit and integration tests will execute and pass.*

### 8. Start the Development Server
Launch the local web server:
```bash
python manage.py runserver
```

Open your browser and navigate to:
👉 **[http://127.0.0.1:8000](http://127.0.0.1:8000)**

---

## 🖥️ Feature Guide

### 1. Customer Profiler (`/`)
- Choose between **Random Forest** or **Decision Tree** engines.
- Click any of the **1-Click Persona Presets** (Executive, Tech Innovator, Household Builder, Prudent Saver) to test different predictions instantly.
- Click **"Segment Customer"** to view real-time results and personalized business playbooks.

### 2. Cohort Analytics (`/analytics/`)
- Visual overview of historical customer submissions.
- **Segment Distribution**: Interactive doughnut chart breaking down proportions of Segment A, B, C, D.
- **Spending Behavior**: Pie chart visualizing Low, Average, and High spend distribution.
- **Algorithm Usage**: Bar chart detailing engine utilization.
- **Top Professions**: Distribution across professions.

### 3. Prediction Archive (`/history/`)
- Search by customer profession.
- Filter records with 1-click segment pills (`All`, `Segment A`, `Segment B`, `Segment C`, `Segment D`).
- Download all customer records to CSV with **"Export to CSV"**.

### 4. Model Diagnostics (`/model-performance/`)
- Inspect real test accuracy, precision, recall, and weighted F1-scores.
- View Random Forest **Feature Importance** rankings showing which attributes most strongly influence segmentation.

---

## 🧠 Machine Learning Pipeline

1. **Dataset**: 8,068 customer records from `data/train.csv` covering demographics, career factors, household size, and spending behavior.
2. **Preprocessing**:
   - Imputation of missing numeric values with medians (`Work_Experience`, `Family_Size`).
   - Categorical one-hot encoding across `Gender`, `Ever_Married`, `Graduated`, `Profession`, `Spending_Score`, `Var_1`.
3. **Algorithms**:
   - **Random Forest Classifier**: 100 estimators, max depth 12, weighted class evaluation.
   - **Decision Tree Classifier**: Pruned 8-level tree for rapid decision rule inspection.
   - **KMeans Clusterer**: Spatial grouping ($k=4$) based on customer Age and Family Size.
4. **Top Predictive Features**:
   - `Age` (~25.2% importance)
   - `Work Experience` (~9.5% importance)
   - `Family Size` (~9.4% importance)
   - `Spending Score` & `Profession` indicators

---

## 🔌 REST API Documentation

The platform includes a programmatic REST API endpoint for automated external integrations.

### Endpoint: Predict Customer Segment
- **URL**: `/api/predict/`
- **Method**: `POST`
- **Headers**: `Content-Type: application/json`

#### Request Payload Example:
```json
{
  "gender": "Male",
  "ever_married": "Yes",
  "age": 42,
  "graduated": "Yes",
  "profession": "Executive",
  "work_experience": 12.0,
  "spending_score": "High",
  "family_size": 2.0,
  "var_1": "Cat_6",
  "prediction_method": "random_forest"
}
```

#### Successful Response (`200 OK`):
```json
{
  "success": true,
  "prediction": {
    "predicted_segmentation": "A",
    "confidence_score": 0.86,
    "prediction_method": "random_forest",
    "cluster_id": 2,
    "processing_time": 0.0182
  },
  "insights": {
    "name": "Segment A - Elite High-Value Customers",
    "persona": "Affluent Professionals & Seasoned Decision Makers",
    "characteristics": "High spending capability, mature career stage, premium brand preference...",
    "marketing_strategy": "Exclusive VIP loyalty programs, invitation-only product previews...",
    "channels": [
      "Private Email Campaigns",
      "Executive Webinars",
      "Concierge Outreach",
      "Premium Print Catalogs"
    ],
    "key_action": "Focus on high-margin upsells and long-term retention incentives."
  }
}
```

---

## ❓ Troubleshooting & FAQ

### Q: `python manage.py runserver` fails with `ModuleNotFoundError`
**Fix**: Ensure your virtual environment is active and all requirements are installed:
```bash
pip install -r requirements.txt
```

### Q: Where are the trained model files stored?
**Fix**: Pre-trained and serialized model files (`random_forest.pkl`, `decision_tree.pkl`, `kmeans.pkl`, `label_encoder.pkl`, `feature_columns.pkl`, `metrics.pkl`) are saved inside the root `models/` directory. If missing, the service automatically retrains them from `data/train.csv`.

### Q: How do I export historical segmentation records?
**Fix**: Go to the **History** tab in the navbar (`/history/`) and click the **"Export to CSV"** button in the top right header.

---

Made with ❤️ using **Django** and **Scikit-Learn**.
