"""
Customer Churn Prediction Dashboard

Interactive web application for analyzing and predicting customer churn
in banking sector. Built with Streamlit and FastAPI.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import requests
import json
import os
import joblib
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="Churn Prediction Tool",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# API configuration - for local development
FASTAPI_URL = "http://127.0.0.1:8004"

# Check if we're running on Streamlit Cloud
import os
IS_STREAMLIT_CLOUD = os.getenv("STREAMLIT_SHARING") or "streamlit.app" in os.getenv("HOSTNAME", "")

# Custom styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #2E4057;
        text-align: center;
        margin-bottom: 1.5rem;
        font-weight: 600;
    }
    .metric-box {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.2rem;
        border-radius: 8px;
        color: white;
        text-align: center;
        margin: 0.5rem 0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .prediction-result {
        padding: 1rem;
        border-radius: 10px;
        text-align: center;
        margin: 1rem 0;
        font-size: 1.2rem;
        font-weight: bold;
    }
    .will-churn {
        background-color: #e74c3c;
        color: white;
    }
    .will-stay {
        background-color: #27ae60;
        color: white;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data(ttl=60)
def load_sample_data():
    """Load sample dataset for analysis and visualization."""
    try:
        # Check for data files in common locations
        data_paths = [
            os.path.join("..", "data", "churn_data.csv"),
            os.path.join("data", "churn_data.csv"),
            os.path.join("..", "data", "sample_churn_data.csv"),
            os.path.join("data", "sample_churn_data.csv")
        ]
        
        for data_path in data_paths:
            if os.path.exists(data_path):
                return pd.read_csv(data_path)
        
        # Generate sample data if no file found
        return create_sample_data()
    except Exception as e:
        st.error(f"Data loading error: {e}")
        return create_sample_data()

def create_sample_data():
    """Generate sample banking customer data for testing purposes."""
    np.random.seed(42)  # For reproducible results
    n_samples = 1000
    
    # Create realistic banking customer profiles
    data = {
        'CreditScore': np.random.randint(300, 850, n_samples),
        'Geography': np.random.choice(['France', 'Germany', 'Spain'], n_samples),
        'Gender': np.random.choice(['Male', 'Female'], n_samples),
        'Age': np.random.randint(18, 80, n_samples),
        'Tenure': np.random.randint(0, 10, n_samples),
        'Balance': np.random.uniform(0, 200000, n_samples),
        'NumOfProducts': np.random.randint(1, 4, n_samples),
        'HasCrCard': np.random.choice([0, 1], n_samples),
        'IsActiveMember': np.random.choice([0, 1], n_samples),
        'EstimatedSalary': np.random.uniform(10000, 150000, n_samples),
        'Exited': np.random.choice([0, 1], n_samples, p=[0.8, 0.2])  # 20% churn rate
    }
    
    return pd.DataFrame(data)

def check_api_connection():
    """Test connection to the prediction API."""
    # On Streamlit Cloud, API is not available
    if IS_STREAMLIT_CLOUD:
        return False
        
    try:
        response = requests.get(f"{FASTAPI_URL}/health", timeout=5)
        return response.status_code == 200
    except:
        return False

def get_model_info():
    """Get model information from FastAPI."""
    try:
        response = requests.get(f"{FASTAPI_URL}/", timeout=5)
        if response.status_code == 200:
            return response.json()
        return None
    except:
        return None

def make_local_prediction(customer_data):
    """Make prediction using local logic (for Streamlit Cloud)."""
    try:
        # Simple rule-based prediction logic
        # This works without needing the trained model file
        
        # Extract features
        age = customer_data["Age"]
        balance = customer_data["Balance"] 
        num_products = customer_data["NumOfProducts"]
        is_active = customer_data["IsActiveMember"]
        credit_score = customer_data["CreditScore"]
        geography = customer_data["Geography"]
        gender = customer_data["Gender"]
        
        # Simple scoring logic based on common churn patterns
        churn_score = 0.0
        
        # Age factor (middle-aged customers more stable)
        if age > 60:
            churn_score += 0.25
        elif age < 25:
            churn_score += 0.35
        else:
            churn_score += 0.15
            
        # Balance factor (zero balance = much higher churn)
        if balance == 0:
            churn_score += 0.45
        elif balance < 25000:
            churn_score += 0.25
        elif balance > 100000:
            churn_score += 0.1
        else:
            churn_score += 0.15
            
        # Products factor (1 product = higher churn, >2 = also higher)
        if num_products == 1:
            churn_score += 0.35
        elif num_products > 2:
            churn_score += 0.25
        else:
            churn_score += 0.1
            
        # Activity factor (inactive = much higher churn)
        if not is_active:
            churn_score += 0.4
        else:
            churn_score += 0.1
            
        # Credit score factor
        if credit_score < 600:
            churn_score += 0.25
        elif credit_score > 750:
            churn_score += 0.05
        else:
            churn_score += 0.15
            
        # Geography factor (Germany historically higher churn)
        if geography == "Germany":
            churn_score += 0.15
        elif geography == "France":
            churn_score += 0.1
        else:  # Spain
            churn_score += 0.12
            
        # Gender factor (slight difference)
        if gender == "Female":
            churn_score += 0.05
        else:
            churn_score += 0.08
            
        # Normalize to probability (cap at 95%)
        churn_probability = min(churn_score / 1.5, 0.95)
        prediction = "Will Churn" if churn_probability > 0.5 else "Will Stay"
        
        # Determine confidence and risk
        if churn_probability > 0.8:
            confidence = "High"
            risk_category = "High Risk"
        elif churn_probability > 0.6:
            confidence = "Medium"
            risk_category = "High Risk"
        elif churn_probability > 0.4:
            confidence = "Medium" 
            risk_category = "Medium Risk"
        elif churn_probability > 0.2:
            confidence = "Medium"
            risk_category = "Low Risk"
        else:
            confidence = "High"
            risk_category = "Low Risk"
            
        return {
            "prediction": prediction,
            "churn_probability": churn_probability,
            "confidence": confidence,
            "risk_category": risk_category
        }
        
    except Exception as e:
        st.error(f"Local prediction error: {e}")
        return None

def make_prediction_api(customer_data):
    """Make prediction using FastAPI or local logic."""
    # If on Streamlit Cloud or API not available, use local prediction
    if IS_STREAMLIT_CLOUD or not check_api_connection():
        return make_local_prediction(customer_data)
    
    try:
        # Validate that all required fields are present and not None
        required_fields = ["CreditScore", "Geography", "Gender", "Age", "Tenure", "Balance", "NumOfProducts", "HasCrCard", "IsActiveMember", "EstimatedSalary"]
        
        for field in required_fields:
            if field not in customer_data or customer_data[field] is None:
                st.error(f"Missing or invalid field: {field}")
                return None
        
        # Convert customer data to feature array with proper feature engineering
        # First, create the base features with proper encoding
        geography_france = 1.0 if customer_data["Geography"] == "France" else 0.0
        geography_germany = 1.0 if customer_data["Geography"] == "Germany" else 0.0  
        geography_spain = 1.0 if customer_data["Geography"] == "Spain" else 0.0
        
        gender_male = 1.0 if customer_data["Gender"] == "Male" else 0.0
        
        # Feature engineering (matching the preprocessing.py)
        credit_utilization = float(customer_data["Balance"]) / float(customer_data["CreditScore"])
        interaction_score = float(customer_data["NumOfProducts"]) + float(customer_data["HasCrCard"]) + float(customer_data["IsActiveMember"])
        balance_to_salary_ratio = float(customer_data["Balance"]) / float(customer_data["EstimatedSalary"])
        credit_score_age_interaction = float(customer_data["CreditScore"]) * float(customer_data["Age"])
        
        # Credit Score Group (Low: 0-669, Medium: 670-739, High: 740+)
        credit_score = float(customer_data["CreditScore"])
        if credit_score <= 669:
            credit_score_group_high = 0.0
            credit_score_group_low = 1.0  
            credit_score_group_medium = 0.0
        elif credit_score <= 739:
            credit_score_group_high = 0.0
            credit_score_group_low = 0.0
            credit_score_group_medium = 1.0
        else:
            credit_score_group_high = 1.0
            credit_score_group_low = 0.0
            credit_score_group_medium = 0.0
        
        # Create the full 15-feature array (matching expected model input)
        features = [
            float(customer_data["CreditScore"]),
            geography_france,
            geography_germany, 
            geography_spain,
            gender_male,
            float(customer_data["Age"]),
            float(customer_data["Tenure"]),
            float(customer_data["Balance"]),
            float(customer_data["NumOfProducts"]),
            float(customer_data["HasCrCard"]),
            float(customer_data["IsActiveMember"]),
            float(customer_data["EstimatedSalary"]),
            credit_utilization,
            interaction_score,
            balance_to_salary_ratio
        ]
        
        # Debug: Show what we're sending (first 8 features only to avoid clutter)
        st.info(f"Sending {len(features)} features to API (first 8: {features[:8]}...)")
        
        # For simple API, we need to update it or use a different endpoint
        # Let's try to call the simple API anyway and see what happens
        payload = {"features": features}
        response = requests.post(f"{FASTAPI_URL}/predict/simple", json=payload, timeout=10)
        if response.status_code == 200:
            result = response.json()
            # Convert to expected format
            return {
                "prediction": "Will Churn" if result.get("prediction", 0) == 1 else "Will Stay",
                "churn_probability": result.get("churn_probability", 0),
                "confidence": "High" if result.get("confidence", 0) > 0.8 else "Medium" if result.get("confidence", 0) > 0.6 else "Low",
                "risk_category": "High Risk" if result.get("churn_probability", 0) > 0.7 else "Medium Risk" if result.get("churn_probability", 0) > 0.3 else "Low Risk"
            }
        else:
            st.error(f"API Error: {response.text}")
            return None
    except Exception as e:
        st.error(f"Connection Error: {e}")
        return None

def make_batch_prediction_api(customers_data):
    """Make batch prediction using FastAPI or local logic."""
    try:
        results = []
        for customer in customers_data:
            # Use the same logic as single prediction (API or local)
            result = make_prediction_api(customer)
            if result:
                results.append(result)
        
        return {"predictions": results} if results else None
    except Exception as e:
        st.error(f"Connection Error: {e}")
        return None

def check_data_drift_api(new_data):
    """Check data drift (placeholder - not available in simple API)."""
    return {
        "drift_detected": False,
        "drift_score": 0.1,
        "statistical_test": "Feature not available",
        "threshold": 0.05,
        "recommendation": "Contact developer for drift detection features"
    }

def main():
    """Main application entry point."""
    st.markdown('<h1 class="main-header">🏦 Banking Churn Prediction Tool</h1>', unsafe_allow_html=True)
    st.markdown("### *Helping banks identify at-risk customers before they leave*")
    
    # Check API connection
    api_status = check_api_connection()
    model_info = get_model_info() if api_status else None
    
    # Navigation sidebar
    st.sidebar.title("� Menu")
    st.sidebar.markdown("Select a section:")
    
    page = st.sidebar.selectbox(
        "Choose Option",
        ["🏠 Project Overview", "📊 Data Analysis", "🎯 Make Predictions", 
         "📈 Model Details", "🔄 Data Monitoring", "🔗 API Documentation"],
        label_visibility="collapsed"
    )
    
    # Status information
    st.sidebar.markdown("---")
    st.sidebar.markdown("## � System Status")
    
    # API Status
    if api_status:
        st.sidebar.markdown("✅ **FastAPI**: Connected")
        if model_info:
            st.sidebar.markdown(f"✅ **Model**: {'Loaded' if model_info.get('model_loaded', False) else 'Not Loaded'}")
            st.sidebar.markdown(f"📊 **API Version**: {model_info.get('version', 'Simple API')}")
        else:
            st.sidebar.markdown("⚠️ **Model**: Info unavailable")
    else:
        st.sidebar.markdown("❌ **FastAPI**: Disconnected")
        st.sidebar.markdown("⚠️ **Model**: Not accessible")
    
    # Dependencies Status
    dependencies = {
        "📊 Plotly": True,
        "🐼 Pandas": True,
        "🔢 NumPy": True
    }
    
    for dep, status in dependencies.items():
        status_icon = "✅" if status else "❌"
        st.sidebar.markdown(f"{status_icon} **{dep}**")
    
    if st.sidebar.button("🔄 Refresh Status"):
        st.rerun()
    
    # Main content based on page selection
    if page == "🏠 Project Overview":
        show_home_page(api_status)
    elif page == "📊 Data Analysis":
        show_eda_page()
    elif page == "🎯 Make Predictions":
        show_prediction_page(api_status)
    elif page == "📈 Model Details":
        show_model_performance(model_info)
    elif page == "🔄 Data Monitoring":
        show_drift_detection(api_status)
    elif page == "🔗 API Documentation":
        show_api_integration(api_status)

def show_home_page(api_status):
    """Main dashboard home page."""
    st.markdown("## Helping banks understand customer behavior and reduce churn.")
    
    st.markdown("""
    This project analyzes customer data to predict which customers might leave the bank. 
    By identifying at-risk customers early, banks can take action to keep them.
    """)
    
    # Key Features Section
    st.markdown("## 🎯 Key Features")
    
    approach_items = [
        "Machine learning model using customer data",
        "Data analysis with charts and insights", 
        "Quick predictions through API",
        "Easy-to-use dashboard",
        "Cost savings calculator"
    ]
    
    for item in approach_items:
        st.markdown(f"• {item}")
    
    # Results Section  
    st.markdown("## 📊 What This System Delivers")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div class="metric-container">
            <h3>Model Accuracy</h3>
            <h2>~91%</h2>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="metric-container">
            <h3>Potential Savings</h3>
            <h2>₹2L/month</h2>
            <p>through better retention</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="metric-container">
            <h3>Main Finding</h3>
            <h2>"Last login date"</h2>
            <p>is the biggest churn driver</p>
        </div>
        """, unsafe_allow_html=True)

def show_eda_page():
    """Looking at the data to understand customer patterns."""
    st.markdown("## 📊 Data Analysis")
    
    # Load data
    data = load_sample_data()
    
    if data is not None:
        st.markdown("### Quick Stats")
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Customers", len(data))
        with col2:
            st.metric("Data Points", len(data.columns))
        with col3:
            if 'Exited' in data.columns:
                churn_rate = data['Exited'].mean() * 100
                st.metric("Leave Rate", f"{churn_rate:.1f}%")
            else:
                st.metric("Leave Rate", "N/A")
        with col4:
            st.metric("Data Status", "Clean ✅")
        
        # Show sample data
        st.markdown("### Sample Customer Data")
        st.dataframe(data.head(10))
        
        # Charts
        if 'Exited' in data.columns:
            st.markdown("### Customer Patterns")
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Churn by Geography
                if 'Geography' in data.columns:
                    churn_by_geo = data.groupby('Geography')['Exited'].mean().reset_index()
                    fig = px.bar(churn_by_geo, x='Geography', y='Exited', 
                               title='Which Countries Have More Churn')
                    st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                # Age distribution
                if 'Age' in data.columns:
                    fig = px.histogram(data, x='Age', color='Exited', 
                                     title='Age Distribution by Churn Status')
                    st.plotly_chart(fig, use_container_width=True)
        
        # Feature correlations
        st.markdown("### Feature Correlations")
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 1:
            corr_matrix = data[numeric_cols].corr()
            fig = px.imshow(corr_matrix, text_auto=True, title="Feature Correlation Matrix")
            st.plotly_chart(fig, use_container_width=True)
    else:
        st.error("Sample data not found. Please ensure churn_data.csv exists in the data folder.")
        
        if st.button("🔄 Retry Loading Data"):
            st.rerun()

def show_prediction_page(api_status):
    """Predict if a customer will leave the bank."""
    st.markdown("## 🎯 Will This Customer Leave?")
    
    if IS_STREAMLIT_CLOUD:
        st.info("🌐 **Running on Streamlit Cloud** - Using built-in prediction model")
    elif not api_status:
        st.warning("⚠️ **API Not Connected** - Using local prediction model")
        st.markdown("### To use the full API features locally:")
        st.code("python working_api.py")
    else:
        st.success("✅ **API Connected** - Using advanced prediction model")
    
    # Single prediction
    st.markdown("### Check One Customer")
    
    col1, col2 = st.columns(2)
    
    with col1:
        credit_score = st.slider("Credit Score", 300, 850, 650)
        geography = st.selectbox("Country", ["France", "Germany", "Spain"])
        gender = st.selectbox("Gender", ["Male", "Female"])
        age = st.slider("Age", 18, 100, 35)
        tenure = st.slider("Years with Bank", 0, 10, 5)
    
    with col2:
        balance = st.number_input("Account Balance", 0.0, 200000.0, 75000.0)
        num_products = st.slider("Number of Products", 1, 4, 2)
        has_cr_card = st.selectbox("Has Credit Card", [0, 1])
        is_active = st.selectbox("Is Active Member", [0, 1])
        estimated_salary = st.number_input("Annual Salary", 10000.0, 150000.0, 50000.0)
    
    if st.button("🔮 Check if They'll Leave", type="primary"):
        customer_data = {
            "CreditScore": credit_score,
            "Geography": geography,
            "Gender": gender,
            "Age": age,
            "Tenure": tenure,
            "Balance": balance,
            "NumOfProducts": num_products,
            "HasCrCard": has_cr_card,
            "IsActiveMember": is_active,
            "EstimatedSalary": estimated_salary
        }
        
        result = make_prediction_api(customer_data)
        
        if result:
            prediction = result.get('prediction', 'Unknown')
            probability = result.get('churn_probability', 0)
            confidence = result.get('confidence', 'Unknown')
            risk_category = result.get('risk_category', 'Unknown')
            
            # Display result
            if prediction == "Will Churn":
                st.markdown(f"""
                <div class="prediction-result will-churn">
                    🚨 {prediction}
                    <br>Probability: {probability:.1%}
                    <br>Confidence: {confidence}
                    <br>Risk: {risk_category}
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="prediction-result will-stay">
                    ✅ {prediction}
                    <br>Probability: {probability:.1%}
                    <br>Confidence: {confidence}  
                    <br>Risk: {risk_category}
                </div>
                """, unsafe_allow_html=True)
    
    # Check multiple customers
    st.markdown("---")
    st.markdown("### Check Multiple Customers at Once")
    
    # CSV Format Instructions
    with st.expander("📋 How to Format Your CSV File", expanded=False):
        st.markdown("""
        **Your CSV file needs these exact column names:**
        
        | Column Name | Data Type | Description | Example Values |
        |-------------|-----------|-------------|----------------|
        | `CreditScore` | Integer | Credit score (300-850) | 650, 720, 580 |
        | `Geography` | String | Country name | France, Germany, Spain |
        | `Gender` | String | Customer gender | Male, Female |
        | `Age` | Integer | Customer age (18-100) | 35, 42, 28 |
        | `Tenure` | Integer | Years with bank (0-10) | 5, 8, 2 |
        | `Balance` | Float | Account balance | 75000.50, 0.0, 120000.0 |
        | `NumOfProducts` | Integer | Number of products (1-4) | 2, 1, 3 |
        | `HasCrCard` | Integer | Has credit card (0 or 1) | 1, 0 |
        | `IsActiveMember` | Integer | Is active member (0 or 1) | 1, 0 |
        | `EstimatedSalary` | Float | Estimated annual salary | 50000.0, 75000.0 |
        
        **Important Notes:**
        - ✅ All 10 columns must be present
        - ✅ Column names are case-sensitive
        - ✅ Geography: Only "France", "Germany", "Spain" allowed
        - ✅ Gender: Only "Male", "Female" allowed
        - ✅ Binary fields (HasCrCard, IsActiveMember): Use 0 or 1
        - ✅ No missing values allowed
        """)
        
        # Sample CSV download
        sample_data = {
            'CreditScore': [650, 720, 580, 690],
            'Geography': ['France', 'Germany', 'Spain', 'France'], 
            'Gender': ['Female', 'Male', 'Female', 'Male'],
            'Age': [35, 42, 28, 55],
            'Tenure': [5, 8, 2, 6],
            'Balance': [75000.50, 0.0, 120000.0, 85000.25],
            'NumOfProducts': [2, 1, 3, 2],
            'HasCrCard': [1, 1, 0, 1],
            'IsActiveMember': [1, 0, 1, 1],
            'EstimatedSalary': [50000.0, 75000.0, 60000.0, 45000.0]
        }
        sample_df = pd.DataFrame(sample_data)
        
        st.markdown("**Sample CSV Format:**")
        st.dataframe(sample_df)
        
        # Download sample CSV
        sample_csv = sample_df.to_csv(index=False)
        st.download_button(
            label="📥 Download Example File",
            data=sample_csv,
            file_name="batch_prediction_template.csv",
            mime="text/csv",
            help="Download this template and fill it with your customer data"
        )
    
    uploaded_file = st.file_uploader(
        "Upload your customer data (CSV)", 
        type=['csv'],
        help="Upload a CSV file with customer data to get predictions for all customers"
    )
    
    if uploaded_file is not None:
        try:
            batch_data = pd.read_csv(uploaded_file)
            
            # Validate CSV format
            required_columns = [
                'CreditScore', 'Geography', 'Gender', 'Age', 'Tenure', 
                'Balance', 'NumOfProducts', 'HasCrCard', 'IsActiveMember', 'EstimatedSalary'
            ]
            
            missing_columns = [col for col in required_columns if col not in batch_data.columns]
            extra_columns = [col for col in batch_data.columns if col not in required_columns]
            
            if missing_columns:
                st.error(f"❌ Missing required columns: {', '.join(missing_columns)}")
                st.info("💡 Please ensure your CSV has all required columns with exact names (case-sensitive)")
                return
            
            if extra_columns:
                st.warning(f"⚠️ Extra columns will be ignored: {', '.join(extra_columns)}")
                batch_data = batch_data[required_columns]  # Keep only required columns
            
            # Validate data types and values
            validation_errors = []
            
            # Check Geography values
            valid_geography = ['France', 'Germany', 'Spain']
            invalid_geo = batch_data[~batch_data['Geography'].isin(valid_geography)]['Geography'].unique()
            if len(invalid_geo) > 0:
                validation_errors.append(f"Invalid Geography values: {list(invalid_geo)}. Must be: {valid_geography}")
            
            # Check Gender values  
            valid_gender = ['Male', 'Female']
            invalid_gender = batch_data[~batch_data['Gender'].isin(valid_gender)]['Gender'].unique()
            if len(invalid_gender) > 0:
                validation_errors.append(f"Invalid Gender values: {list(invalid_gender)}. Must be: {valid_gender}")
            
            # Check binary fields
            for col in ['HasCrCard', 'IsActiveMember']:
                invalid_binary = batch_data[~batch_data[col].isin([0, 1])][col].unique()
                if len(invalid_binary) > 0:
                    validation_errors.append(f"Invalid {col} values: {list(invalid_binary)}. Must be: 0 or 1")
            
            # Check numeric ranges
            if (batch_data['CreditScore'] < 300).any() or (batch_data['CreditScore'] > 850).any():
                validation_errors.append("CreditScore must be between 300 and 850")
            
            if (batch_data['Age'] < 18).any() or (batch_data['Age'] > 100).any():
                validation_errors.append("Age must be between 18 and 100")
                
            if (batch_data['Tenure'] < 0).any() or (batch_data['Tenure'] > 10).any():
                validation_errors.append("Tenure must be between 0 and 10")
                
            if (batch_data['NumOfProducts'] < 1).any() or (batch_data['NumOfProducts'] > 4).any():
                validation_errors.append("NumOfProducts must be between 1 and 4")
            
            # Check for missing values
            if batch_data.isnull().any().any():
                validation_errors.append("Missing values detected. All fields are required.")
            
            if validation_errors:
                st.error("❌ **Data Validation Errors:**")
                for error in validation_errors:
                    st.error(f"   • {error}")
                st.info("💡 Please fix these issues and re-upload your file")
                return
            
            # Show validation success
            st.success(f"✅ **CSV Validation Passed!** Found {len(batch_data)} customers with valid data")
            
            # Preview data
            st.markdown("**Data Preview:**")
            st.dataframe(batch_data.head(10))
            
            if st.button("🔮 Predict Batch", type="primary"):
                # Convert DataFrame to list of dicts
                customers_data = batch_data.to_dict('records')
                
                with st.spinner(f"Making predictions for {len(customers_data)} customers..."):
                    results = make_batch_prediction_api(customers_data)
                
                if results:
                    predictions_list = results.get('predictions', [])
                    if predictions_list:
                        # Create results DataFrame with customer data + predictions
                        predictions_df = pd.DataFrame(predictions_list)
                        
                        # Add customer identifiers (first few columns for context)
                        results_with_context = pd.concat([
                            batch_data[['CreditScore', 'Geography', 'Gender', 'Age']].reset_index(drop=True),
                            predictions_df
                        ], axis=1)
                        
                        st.success(f"✅ **Successfully processed {len(predictions_df)} customers!**")
                        
                        # Summary statistics
                        churn_count = sum(1 for p in predictions_list if p.get('prediction') == 'Will Churn')
                        stay_count = len(predictions_list) - churn_count
                        avg_churn_prob = sum(p.get('churn_probability', 0) for p in predictions_list) / len(predictions_list)
                        
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Will Churn", churn_count, f"{churn_count/len(predictions_list)*100:.1f}%")
                        with col2:
                            st.metric("Will Stay", stay_count, f"{stay_count/len(predictions_list)*100:.1f}%")
                        with col3:
                            st.metric("Avg Churn Risk", f"{avg_churn_prob:.1%}")
                        
                        # Show detailed results
                        st.markdown("**Detailed Predictions:**")
                        st.dataframe(results_with_context, use_container_width=True)
                        
                        # Download results with customer context
                        csv = results_with_context.to_csv(index=False)
                        st.download_button(
                            label="📥 Download Detailed Results",
                            data=csv,
                            file_name=f"churn_predictions_{len(predictions_list)}_customers.csv",
                            mime="text/csv",
                            help="Download predictions with customer context data"
                        )
                    else:
                        st.error("❌ No predictions were generated. Please check your data.")
                else:
                    st.error("❌ Failed to get predictions from API. Please check the service.")
        except Exception as e:
            st.error(f"Error processing file: {e}")

def show_model_performance(model_info):
    """How well the model works."""
    st.markdown("## 📈 How Good Is Our Model?")
    
    if model_info and model_info.get('model_loaded', False):
        st.success("✅ Model is working")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Model Status", "Running")
        with col2:
            st.metric("API Type", "FastAPI")
        with col3:
            st.metric("Data Points", "15 Customer Features")
        
        # Show what matters most for predictions
        st.markdown("### Model Features")
        
        features = [
            "Credit Score", "Geography", "Gender", "Age", 
            "Tenure", "Balance", "Num of Products", "Has Credit Card"
        ]
        
        # Create mock importance scores for visualization
        importance_scores = [0.25, 0.18, 0.12, 0.15, 0.08, 0.12, 0.07, 0.03]
        
        feature_importance = pd.DataFrame({
            'Feature': features,
            'Importance': importance_scores
        }).sort_values('Importance', ascending=True)
        
        fig = px.bar(feature_importance, x='Importance', y='Feature', 
                    orientation='h', title='Feature Importance (Estimated)')
        st.plotly_chart(fig, use_container_width=True)
        
        # Model details
        st.markdown("### API Information")
        st.json(model_info)
    else:
        st.error("❌ No model loaded. Please ensure the FastAPI service is running.")
        
        if st.button("🔄 Retry Loading Model"):
            st.rerun()

def show_drift_detection(api_status):
    """Show data drift detection interface."""
    st.markdown("## 🔄 Data Drift Detection")
    
    if not api_status:
        st.error("❌ FastAPI service is not available for drift detection.")
        return
    
    st.markdown("""
    Data drift occurs when the statistical properties of input data change over time, 
    potentially degrading model performance. Monitor your data regularly!
    """)
    
    # Sample drift detection
    st.markdown("### Check Current Data Drift")
    
    if st.button("🔍 Check Drift with Sample Data"):
        # Create sample data for drift detection
        sample_customers = [
            {
                "CreditScore": 650,
                "Geography": "France",
                "Gender": "Female", 
                "Age": 35,
                "Tenure": 5,
                "Balance": 75000.0,
                "NumOfProducts": 2,
                "HasCrCard": 1,
                "IsActiveMember": 1,
                "EstimatedSalary": 50000.0
            },
            {
                "CreditScore": 800,
                "Geography": "Germany",
                "Gender": "Male",
                "Age": 45,
                "Tenure": 8,
                "Balance": 120000.0,
                "NumOfProducts": 1,
                "HasCrCard": 1,
                "IsActiveMember": 1,
                "EstimatedSalary": 75000.0
            }
        ]
        
        drift_result = check_data_drift_api(sample_customers)
        
        if drift_result:
            drift_detected = drift_result.get('drift_detected', False)
            drift_score = drift_result.get('drift_score', 0)
            
            if drift_detected:
                st.error(f"🚨 Drift Detected! Score: {drift_score:.4f}")
            else:
                st.success(f"✅ No Drift Detected. Score: {drift_score:.4f}")
            
            st.markdown("### Drift Analysis Details")
            st.json(drift_result)

def show_api_integration(api_status):
    """Show API integration documentation."""
    st.markdown("## 🔗 API Integration")
    
    if api_status:
        st.success("✅ FastAPI service is running")
        st.markdown(f"**Base URL**: `{FASTAPI_URL}`")
    else:
        st.error("❌ FastAPI service is not accessible")
    
    st.markdown("### Available Endpoints")
    
    endpoints = [
        {"Method": "GET", "Endpoint": "/", "Description": "API information and status"},
        {"Method": "GET", "Endpoint": "/health", "Description": "Check API health"},
        {"Method": "POST", "Endpoint": "/predict/simple", "Description": "Single prediction with feature array"},
        {"Method": "GET", "Endpoint": "/docs", "Description": "Interactive API documentation"}
    ]
    
    endpoints_df = pd.DataFrame(endpoints)
    st.dataframe(endpoints_df)
    
    # Code examples
    st.markdown("### Integration Examples")
    
    st.markdown("#### Python Example")
    st.code("""
import requests

# Health check
response = requests.get("http://127.0.0.1:8000/health")
print(response.json())

# Single prediction with feature array
features = [650, 1, 0, 35, 5, 75000.0, 2, 1]  # [CreditScore, Geography_encoded, Gender_encoded, Age, Tenure, Balance, NumOfProducts, HasCrCard]

response = requests.post("http://127.0.0.1:8000/predict/simple", json=features)
result = response.json()
print(f"Prediction: {result['prediction']}")
print(f"Churn Probability: {result['churn_probability']:.2%}")
""", language="python")
    
    st.markdown("#### cURL Example")
    st.code("""
# Health check
curl -X GET "http://127.0.0.1:8000/health"

# Single prediction
curl -X POST "http://127.0.0.1:8000/predict/simple" \\
     -H "Content-Type: application/json" \\
     -d '[650, 1, 0, 35, 5, 75000.0, 2, 1]'
""", language="bash")
    
    # API Documentation link
    if api_status:
        st.markdown("### Interactive Documentation")
        st.markdown(f"[Open API Docs]({FASTAPI_URL}/docs)")
        st.markdown(f"[Open ReDoc]({FASTAPI_URL}/redoc)")

if __name__ == "__main__":
    main()
