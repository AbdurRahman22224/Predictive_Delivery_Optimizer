"""
Streamlit Dashboard for Predictive Delivery Optimizer
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import joblib
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Page config
st.set_page_config(
    page_title="Predictive Delivery Optimizer",
    page_icon="📦",
    layout="wide"
)

# Sidebar
st.sidebar.title("Navigation")
page = st.sidebar.selectbox("Select Page", ["Overview", "Model Performance", "Predictions", "Insights"])

# Load data (cached)
@st.cache_data
def load_data():
    """Load processed data"""
    try:
        df = pd.read_csv("processed/features.csv")
        return df
    except FileNotFoundError:
        st.error("Data file not found. Please run data_prep.py first.")
        return None

@st.cache_resource
def load_models():
    """Load trained models"""
    models = {}
    try:
        from catboost import CatBoostClassifier, CatBoostRegressor
        classifier = CatBoostClassifier()
        classifier.load_model("models/best_catboost_classifier.cbm")
        regressor = CatBoostRegressor()
        regressor.load_model("models/best_catboost_regressor.cbm")
        preprocessor = joblib.load("processed/preprocessor.joblib")
        models['classifier'] = classifier
        models['regressor'] = regressor
        models['preprocessor'] = preprocessor
    except Exception as e:
        st.warning(f"Models not loaded: {e}")
    return models

# Load data
df = load_data()

if df is None:
    st.stop()

# Filters in sidebar
st.sidebar.header("Filters")
carriers = st.sidebar.multiselect("Carriers", df['Carrier'].unique())
priorities = st.sidebar.multiselect("Priority", df['Priority'].unique())
categories = st.sidebar.multiselect("Product Category", df['Product_Category'].unique())

# Apply filters
df_filtered = df.copy()
if carriers:
    df_filtered = df_filtered[df_filtered['Carrier'].isin(carriers)]
if priorities:
    df_filtered = df_filtered[df_filtered['Priority'].isin(priorities)]
if categories:
    df_filtered = df_filtered[df_filtered['Product_Category'].isin(categories)]

if page == "Overview":
    st.header("📊 Overview Dashboard")
    
    # KPI Cards
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        st.metric("Total Orders", len(df_filtered))
    
    with col2:
        delay_pct = df_filtered['is_delayed'].mean() * 100
        st.metric("Delay Rate", f"{delay_pct:.1f}%")
    
    with col3:
        avg_delay = df_filtered['delay_days'].mean()
        st.metric("Avg Delay Days", f"{avg_delay:.1f}")
    
    with col4:
        avg_cost = df_filtered['Delivery_Cost_INR'].mean()
        st.metric("Avg Cost (INR)", f"{avg_cost:.0f}")
    
    with col5:
        avg_rating = df_filtered['Customer_Rating'].mean()
        st.metric("Avg Rating", f"{avg_rating:.1f}/5")
    
    # Charts
    st.subheader("Delay Rate by Carrier")
    carrier_delay = df_filtered.groupby('Carrier').agg({
        'is_delayed': ['sum', 'count']
    }).reset_index()
    carrier_delay.columns = ['Carrier', 'Delayed', 'Total']
    carrier_delay['Delay_Rate'] = (carrier_delay['Delayed'] / carrier_delay['Total'] * 100).round(1)
    
    fig = px.bar(carrier_delay, x='Carrier', y='Delay_Rate', 
                 title="Delay Rate by Carrier (%)", color='Carrier',  # adds matching bar colors
                 color_discrete_sequence=px.colors.sequential.Tealgrn,  # unified color palette  
                 template="plotly_white")
    st.plotly_chart(fig, use_container_width=True)
    
    # Distance vs Cost
    st.subheader("Distance vs Delivery Cost")
    fig = px.scatter( df_filtered, x='Distance_KM', y='Delivery_Cost_INR',
                 color='is_delayed', title="Distance vs Cost",
                 labels={'is_delayed': 'Delayed', 'Delivery_Cost_INR': 'Cost (INR)'},
                 )
    st.plotly_chart(fig, use_container_width=True)

    # Priority Distribution (Pie Chart)
    st.subheader("Order Priority Distribution")
    priority_dist = df_filtered['Priority'].value_counts()
    fig = px.pie(values=priority_dist.values, names=priority_dist.index, 
                 title="Order Priority Distribution",
                 color_discrete_sequence=px.colors.sequential.YlOrBr,
                 hole=0.4)
    fig.update_traces(textposition='inside', textinfo='percent+label')
    st.plotly_chart(fig, use_container_width=True)

    # Product Category Delay Rates (Horizontal Bar Chart)
    st.subheader("Delay Rate by Product Category")
    category_delay = df_filtered.groupby('Product_Category').agg({
        'is_delayed': 'mean',
        'Order_ID': 'count'
    }).reset_index()
    category_delay.columns = ['Product_Category', 'Delay_Rate', 'Order_Count']
    category_delay['Delay_Rate'] = (category_delay['Delay_Rate'] * 100).round(1)
    category_delay = category_delay.sort_values('Delay_Rate', ascending=False)
    
    fig = px.bar(category_delay, x='Delay_Rate', y='Product_Category', orientation='h',
                 color='Product_Category', 
                 title="Delay Rate by Product Category (%)",
                 labels={'Delay_Rate': 'Delay Rate (%)', 'Product_Category': 'Product Category'},
                 color_discrete_sequence=px.colors.qualitative.Set3)
    fig.update_layout(yaxis={'categoryorder':'total descending'}, showlegend=False)
    st.plotly_chart(fig, use_container_width=True)
    
    # Correlation Heatmap
    st.subheader("Feature Correlation Heatmap")
    numeric_cols = df_filtered.select_dtypes(include=[np.number]).columns
    corr = df_filtered[numeric_cols].corr()
    fig = px.imshow(corr, text_auto=True, aspect="auto", title="Correlation Matrix", 
    color_continuous_scale='Tealgrn', template="plotly_white")
    st.plotly_chart(fig, use_container_width=True)

elif page == "Model Performance":
    st.header("🤖 Model Performance")
    
    # Check if models exist
    model_files = Path("models").glob("*.cbm")
    if not list(model_files):
        st.error("No trained models found. Please run modeling.py first.")
        st.stop()
    
    # Load model info
    try:
        import json
        with open("models/best_classifier_info.json", 'r') as f:
            classifier_info = json.load(f)
        
        st.subheader("Best Classification Model")
        st.json(classifier_info)
        
        # Metrics
        if 'metrics' in classifier_info:
            metrics = classifier_info['metrics']
            col1, col2, col3, col4, col5 = st.columns(5)
            with col1:
                st.metric("Accuracy", f"{metrics.get('accuracy', 0):.4f}")
            with col2:
                st.metric("Precision", f"{metrics.get('precision', 0):.4f}")
            with col3:
                st.metric("Recall", f"{metrics.get('recall', 0):.4f}")
            with col4:
                st.metric("F1-Score", f"{metrics.get('f1_score', 0):.4f}")
            with col5:
                st.metric("ROC-AUC", f"{metrics.get('roc_auc', 0):.4f}")
    except Exception as e:
        st.warning(f"Could not load model info: {e}")
    
    # Load regressor info
    try:
        with open("models/best_regressor_info.json", 'r') as f:
            regressor_info = json.load(f)
        
        st.subheader("Best Regression Model")
        st.json(regressor_info)
        
        if 'metrics' in regressor_info:
            metrics = regressor_info['metrics']
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("RMSE", f"{metrics.get('rmse', 0):.2f} days")
            with col2:
                st.metric("MAE", f"{metrics.get('mae', 0):.2f} days")
            with col3:
                st.metric("R²", f"{metrics.get('r2', 0):.4f}")
    except Exception as e:
        st.warning(f"Could not load regressor info: {e}")
    

elif page == "Predictions":
    st.header("🔮 Predictions")
    
    models = load_models()
    
    if 'classifier' not in models:
        st.error("Models not loaded. Please train models first using modeling.py")
        st.stop()
    
    tab1, = st.tabs(["Single Prediction"])
    
    with tab1:
        st.subheader("Single Order Prediction")
        
        col1, col2 = st.columns(2)
        
        with col1:
            carrier = st.selectbox("Carrier", df['Carrier'].unique())
            priority = st.selectbox("Priority", df['Priority'].unique())
            product_category = st.selectbox("Product Category", df['Product_Category'].unique())
            distance_km = st.number_input("Distance (KM)", min_value=0.0, value=500.0)
            order_value = st.number_input("Order Value (INR)", min_value=0.0, value=1000.0)
        
        with col2:
            delivery_cost = st.number_input("Delivery Cost (INR)", min_value=0.0, value=500.0)
            promised_days = st.number_input("Promised Delivery Days", min_value=1, value=3, step=1)
            weather_impact = st.selectbox("Weather Impact", ['Unknown', 'Fog', 'Light_Rain', 'Heavy_Rain'])
            fuel_consumption = st.number_input("Fuel Consumption (L)", min_value=0.0, value=50.0)
            toll_charges = st.number_input("Toll Charges (INR)", min_value=0.0, value=100.0)
        
        if st.button("Predict", type="primary"):
            # Create input dataframe
            input_data = pd.DataFrame({
                'Order_ID': ['PRED001'],
                'Priority': [priority],
                'Product_Category': [product_category],
                'Order_Value_INR': [order_value],
                'Carrier': [carrier],
                'Promised_Delivery_Days': [promised_days],
                'Delivery_Cost_INR': [delivery_cost],  # Estimate
                'Distance_KM': [distance_km],
                'Fuel_Consumption_L': [fuel_consumption],
                'Toll_Charges_INR': [toll_charges],
                'Weather_Impact': [weather_impact]
            })
            
            # Add derived features
            input_data['cost_per_promised_day'] = input_data['Delivery_Cost_INR'] / input_data['Promised_Delivery_Days']
            
            # Buckets
            input_data['promised_days_bucket'] = '0-1' if promised_days <= 1 else ('2-3' if promised_days <= 3 else '4+')
            input_data['distance_bucket'] = 'short' if distance_km < 500 else ('medium' if distance_km < 1500 else 'long')
            input_data['order_value_bucket'] = 'Q1' if order_value < df['Order_Value_INR'].quantile(0.25) else (
                'Q2' if order_value < df['Order_Value_INR'].quantile(0.5) else (
                    'Q3' if order_value < df['Order_Value_INR'].quantile(0.75) else 'Q4'
                )
            )
            
            # Preprocess
            X = input_data.drop(columns=['Order_ID'])
            
            # Ensure all features are in the correct order expected by preprocessor
            # The preprocessor expects these features (without Customer_Rating and Traffic_Delay_Minutes)
            expected_features = ['Priority', 'Product_Category', 'Order_Value_INR', 'Carrier', 
                               'Promised_Delivery_Days', 'Delivery_Cost_INR', 'Distance_KM', 
                               'Fuel_Consumption_L', 'Toll_Charges_INR', 'Weather_Impact',
                               'cost_per_promised_day', 'promised_days_bucket', 'order_value_bucket', 'distance_bucket']
            
            # Reorder columns to match preprocessor expectations
            X = X[expected_features]
            
            X_trans = models['preprocessor'].transform(X)
            
            # Predict
            delay_prob = models['classifier'].predict_proba(X_trans)[0, 1]
            delay_days = max(0, models['regressor'].predict(X_trans)[0])
            
            # Display results
            st.success("Prediction Complete!")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.metric("Delay Probability", f"{delay_prob*100:.1f}%")
                if delay_prob < 0.3:
                    risk = "LOW"
                    color = "🟢"
                elif delay_prob < 0.6:
                    risk = "MEDIUM"
                    color = "🟡"
                else:
                    risk = "HIGH"
                    color = "🔴"
                st.metric("Risk Level", f"{color} {risk}")
            
            with col2:
                st.metric("Expected Delay Days", f"{delay_days:.1f}")
    
    # with tab2:
    #     st.subheader("Batch Prediction")
    #     st.info("Upload a CSV file with order data (columns must match training features)")
        
    #     uploaded_file = st.file_uploader("Upload CSV", type="csv")
        
    #     if uploaded_file:
    #         try:
    #             batch_df = pd.read_csv(uploaded_file)
    #             st.dataframe(batch_df.head())
                
    #             if st.button("Process Batch"):
    #                 st.success("Batch processing feature coming soon!")
    #                 st.download_button("Download Predictions", data="", file_name="predictions.csv")
    #         except Exception as e:
    #             st.error(f"Error reading file: {e}")

elif page == "Insights":
    st.header("💡 Business Insights")
    
    # Carrier Rankings
    st.subheader("Carrier Rankings")
    
    carrier_stats = df_filtered.groupby('Carrier').agg({
        'is_delayed': 'mean',
        'Delivery_Cost_INR': 'mean',
        'Customer_Rating': 'mean'
    }).reset_index()
    carrier_stats.columns = ['Carrier', 'Delay_Rate', 'Avg_Cost', 'Avg_Rating']
    carrier_stats = carrier_stats.sort_values('Delay_Rate')
    
    st.dataframe(carrier_stats, use_container_width=True)
    
    # Actionable Rules
    st.subheader("Actionable Recommendations")
    
    # Example rules based on data
    st.markdown("""
    **Key Findings:**
    - Express orders with distances > 1500 km have higher delay rates
    - Certain carriers show consistent on-time performance
    - Heavy rain and fog significantly impact delivery times
    - Orders with low promised delivery days (1-2) are more likely to be delayed
    
    **Recommendations:**
    1. Consider alternative carriers for long-distance orders
    2. Adjust promised delivery days based on weather forecasts
    3. Allocate extra buffer time for fog and rain conditions
    4. Monitor carriers with delay rates > 50%
    """)

# Footer
st.sidebar.markdown("---")
st.sidebar.markdown("**Predictive Delivery Optimizer**")
st.sidebar.markdown("NexGen Logistics")
