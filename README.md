# 🏦 Customer Churn Prediction System

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://customerchurnpredictionanalysis.streamlit.app/)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

> **Banking Churn Prediction Tool** - Helping banks identify at-risk customers before they leave

## 🎯 Project Overview

This project analyzes customer data to predict which customers might leave the bank. By identifying at-risk customers early, banks can take action to keep them and improve customer retention.

## 🏆 Key Achievements

• **Modelled churn risk for 10K customers**, tuning Random Forest to **86% accuracy & 0.85 ROC-AUC score**

• **Increased accuracy by 15% & ROC-AUC by 8.8%** over the baseline logistic regression model used initially

• **Found customers aged 51-60 have 2.8x higher churn risk**; targeting pre-retirement financial planning can lower attrition and boost retention revenue

• **Deployed Streamlit app** with batch uploads, **interactive churn probability gauges** and exportable reports for analysis

## ✨ Key Features

- **Machine learning model** using customer data
- **Data analysis** with charts and insights  
- **Interactive prediction interface** with visual probability gauges
- **Batch processing** for multiple customers at once
- **Easy-to-use dashboard** with professional visualizations
- **Export capabilities** for business reporting

## 🚀 Live Demo

**Try it here:** [Customer Churn Prediction Tool](https://customerchurnpredictionanalysis.streamlit.app/)

## 📊 Model Performance

| Metric | Baseline (Logistic Regression) | Final Model (Random Forest) | Improvement |
|--------|--------------------------------|------------------------------|-------------|
| **Accuracy** | 70.4% | 86.2% | +15.8% |
| **ROC-AUC** | 0.764 | 0.852 | +8.8% |
| **Precision** | 0.55 | 0.78 | +23% |
| **Recall** | 0.47 | 0.72 | +25% |

## 🎛️ Interactive Features

### Visual Probability Gauges
- **Real-time churn probability gauges** (0-100%)
- **Color-coded risk zones**: Green (Low), Yellow (Medium), Red (High)
- **Interactive dashboard** with multiple metrics

### Batch Analysis Dashboard
- **Churn Rate Gauge**: Overall percentage of at-risk customers
- **Retention Rate Gauge**: Percentage likely to stay
- **Average Risk Gauge**: Mean probability across all customers

## 📈 Business Insights

### 🎯 High-Risk Customer Segments
- **Age 51-60**: 2.8x higher churn risk (56% churn rate)
- **Single product customers**: 27% churn rate vs 8% for multi-product
- **Inactive members**: 27% churn rate vs 14% for active members
- **Germany customers**: 32% churn rate (highest by geography)

### 💡 Retention Strategies
1. **Pre-retirement planning services** for 51-60 age group
2. **Product expansion** campaigns for single-product customers  
3. **Engagement programs** for inactive members
4. **Localized retention** strategies for German market

## 🛠️ Technical Stack

- **Python 3.8+**: Core programming language
- **Scikit-learn**: Machine learning models
- **Pandas & NumPy**: Data manipulation
- **Plotly**: Interactive visualizations and gauges
- **Streamlit**: Web application framework
- **FastAPI**: RESTful API (optional)

## 📁 Project Structure

```
CustomerChurnFireProject/
├── app/
│   └── streamlit_app.py          # Main Streamlit application
├── data/
│   └── Churn_Modelling.csv       # Customer dataset (10K records)
├── models/
│   └── churn_model.pkl           # Trained Random Forest model
├── notebooks/
│   └── eda-and-modeling.ipynb    # Analysis and model development
├── tests/
│   └── test_api.py               # API endpoint tests
└── requirements.txt              # Python dependencies
```

## 🚀 Quick Start

### Option 1: Use the Live App
Visit the deployed application: [Customer Churn Prediction Tool](https://customerchurnpredictionanalysis.streamlit.app/)

### Option 2: Run Locally

```bash
# Clone the repository
git clone https://github.com/omprakash-mourya/customer-churn-analysis.git
cd customer-churn-analysis/CustomerChurnFireProject

# Install dependencies
pip install -r requirements.txt

# Run the Streamlit app
streamlit run app/streamlit_app.py
```

The app will open in your browser at `http://localhost:8501`

## 📊 How to Use

### Single Customer Prediction
1. Navigate to "🎯 Make Predictions"
2. Enter customer details in the form
3. Click "🔮 Predict Churn"
4. View the **interactive probability gauge** and risk assessment

### Batch Predictions
1. Go to "🎯 Make Predictions" 
2. Download the CSV template
3. Upload your customer data file
4. View the **multi-gauge dashboard** with:
   - Overall churn rate
   - Retention rate  
   - Average risk score
5. Download detailed results

### Data Analysis
1. Visit "📊 Data Analysis" to explore:
   - Customer demographics
   - Churn patterns by age, geography, products
   - Feature importance analysis

## 🧪 Testing

The project includes comprehensive test suites:

```bash
# Run API tests
pytest tests/test_api.py -v

# Test model validation
python -m pytest tests/ --cov=app
```

## 📈 Model Details

### Feature Engineering
- **15 engineered features** from 10 original columns
- **One-hot encoding** for categorical variables
- **Interaction features** (CreditScore × Age)
- **Ratio features** (Balance/Salary)

### Model Selection
- Tested: Logistic Regression, Random Forest, XGBoost
- **Best performer**: Random Forest with hyperparameter tuning
- **Cross-validation**: 5-fold CV for robust evaluation

## 📧 Contact

**Omprakash Mourya**
- Email: ommourya2006@gmail.com
- GitHub: [@omprakash-mourya](https://github.com/omprakash-mourya)
- Project: [Customer Churn Analysis](https://github.com/omprakash-mourya/customer-churn-analysis)
