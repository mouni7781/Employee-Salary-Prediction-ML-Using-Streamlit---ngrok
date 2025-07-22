import streamlit as st
import pandas as pd
import joblib
import numpy as np
from datetime import datetime
import io

st.set_page_config(page_title="Advanced Salary Prediction", page_icon="💼", layout="wide")
model = joblib.load(r'C:/Users/mouni/OneDrive/Desktop/Resources/Pyton-AI/Project/files/best_model.pkl')
columns = joblib.load(r'C:/Users/mouni/OneDrive/Desktop/Resources/Pyton-AI/Project/files/columns.pkl')

st.markdown("""
<style>
    .reportview-container { background: #20232A; }
    .main .block-container {
        padding: 2rem;
        background-color: #343A40; /* Darker background for main content */
        border-radius: 8px;
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.3);
    }
    .stApp { background-color: #20232A; }
    .css-1d391kg { /* Sidebar */ background-color: #ffffff; border-right: 1px solid #e6e6e6; padding: 1rem; }
    h1 { color: #2e86de; text-align: center; margin-bottom: 0.5rem; }
    h2, h3 { color: #ffffff; /* Adjust for dark background */ margin-top: 1.5rem; }
    .stButton>button {
        background-color: #28a745; color: white; border-radius: 5px; border: none;
        padding: 0.6rem 1.2rem; font-size: 1rem; transition: all 0.2s ease-in-out;
    }
    .stButton>button:hover { background-color: #218838; transform: translateY(-2px); }
    .stTextInput>div>div>input, .stSelectbox>div>div>select, .stSlider>div>div>div>div {
        border-radius: 5px; border: 1px solid #ced4da; padding: 0.375rem 0.75rem;
    }
    .stAlert { border-radius: 5px; }
    /* Adjust text color for white content within dark main block */
    .stMarkdown, .stText, .stLabel, .stBlock { color: white !important; }
    .css-r69unv { color: white !important; } /* specifically for default text within columns */
</style>
""", unsafe_allow_html=True)

st.title("Employee Salary Prediction Application")
st.markdown("Predict whether an individual's annual income exceeds $50K using detailed demographic and employment features. "
            "You can enter data manually or upload a CSV file to predict multiple records at once.")

st.sidebar.header("⚙️ Choose Input Mode")
input_mode = st.sidebar.radio("Select input mode:", ["Single Input", "Batch CSV Upload"])

OPTIONS = {
    'workclass': ['Private', 'Self-emp-not-inc', 'Local-gov', 'State-gov', 'Self-emp-inc', 'Federal-gov', 'NotListed', 'Without-pay', 'Never-worked'],
    'marital_status': ['Married-civ-spouse', 'Never-married', 'Divorced', 'Separated', 'Widowed', 'Married-spouse-absent', 'Married-AF-spouse'],
    'occupation': ['Prof-specialty', 'Craft-repair', 'Exec-managerial', 'Adm-clerical', 'Sales', 'Other-service', 'Machine-op-inspct', 'Transport-moving', 'Handlers-cleaners', 'Farming-fishing', 'Tech-support', 'Protective-serv', 'Priv-house-serv', 'Armed-Forces', 'No Information'],
    'relationship': ['Husband', 'Not-in-family', 'Own-child', 'Unmarried', 'Wife', 'Other-relative'],
    'race': ['White', 'Black', 'Asian-Pac-Islander', 'Amer-Indian-Eskimo', 'NotReported'],
    'gender': ['Male', 'Female'],
    'native_country': ['United-States', 'Mexico', 'Philippines', 'Germany', 'Canada', 'El-Salvador', 'India', 'Other']
}

def preprocess_input(df):
    df_processed = df.copy()
    for col in ['capital-gain', 'capital-loss']:
        if col in df_processed.columns and pd.api.types.is_numeric_dtype(df_processed[col]):
            df_processed[col] = np.log1p(df_processed[col])
    
    encoded = pd.get_dummies(df_processed)
    for col in columns:
        if col not in encoded.columns:
            encoded[col] = 0
    return encoded[columns]


if input_mode == "Single Input":
    st.subheader("📊 Enter Individual's Details")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("### 👤 Personal Information")
        # Minor optimization: put arguments on one line if they're short
        age = st.number_input("Age", 17, 74, 30, help="Age of the individual.")
        gender = st.selectbox("Gender", OPTIONS['gender'], help="Gender of the individual.")
        race = st.selectbox("Race", OPTIONS['race'], help="Racial background.")
        native_country = st.selectbox("Native Country", OPTIONS['native_country'], help="Country of origin.")

    with col2:
        st.markdown("### 💼 Employment Details")
        workclass = st.selectbox("Workclass", OPTIONS['workclass'], help="Type of employer.")
        occupation = st.selectbox("Occupation", OPTIONS['occupation'], help="Individual's occupation.")
        hours_per_week = st.slider("Hours per week", 1, 99, 40, help="Average hours worked per week.")
        education_num = st.slider("Education Level (numeric)", 3, 16, 10, help="Years of education completed.")

    with col3:
        st.markdown("### 💰 Financial & Marital Status")
        marital_status = st.selectbox("Marital Status", OPTIONS['marital_status'], help="Marital status of the individual.")
        relationship = st.selectbox("Relationship", OPTIONS['relationship'], help="Relationship status within family.")
        capital_gain = st.number_input("Capital Gain", 0, None, 0, help="Capital gains from investments.") # 'None' for max_value if unbounded
        capital_loss = st.number_input("Capital Loss", 0, None, 0, help="Capital losses from investments.") # 'None' for max_value if unbounded

    # Data collection is already concise
    input_data = {
        'age': age, 'workclass': workclass, 'educational-num': education_num,
        'marital-status': marital_status, 'occupation': occupation,
        'relationship': relationship, 'race': race, 'gender': gender,
        'capital-gain': capital_gain, 'capital-loss': capital_loss,
        'hours-per-week': hours_per_week, 'native-country': native_country
    }
    input_df = pd.DataFrame([input_data]) 

    st.markdown("---")
    col_btn1, col_btn2, col_btn3 = st.columns([1, 1, 1])
    with col_btn2:
        if st.button("🚀 Predict Income Category", use_container_width=True):
            try:
                processed_input = preprocess_input(input_df)
                prediction = model.predict(processed_input)[0]
                proba_over_50k = model.predict_proba(processed_input)[0][1] # Rename for clarity
                
                st.subheader("Prediction Result")
                if prediction == 1:
                    st.success(f"🎉 **Estimated Income: >$50K** with probability **{proba_over_50k:.2f}**")
                    st.balloons()
                else:
                    st.info(f"📉 **Estimated Income: <=$50K** with probability **{1 - proba_over_50k:.2f}**")
            except Exception as e:
                st.error(f"⚠️ Prediction failed. Details: {e}")

    st.markdown("---")
    with st.expander("Show Raw Input Features"):
        st.dataframe(input_df)
else:  
    st.subheader("⬆️ Upload CSV for Batch Prediction")
    uploaded_file = st.file_uploader("Drag and drop your CSV file here or click to browse", type=["csv"],
                                     help="Upload a CSV file containing rows of features for prediction. "
                                          "Ensure column names match the single input mode.")
    
    if uploaded_file: 
        try:
            batch_df = pd.read_csv(uploaded_file)
            st.write("📂 **Uploaded Data Sample:**", batch_df.head())
            
            with st.spinner("Processing and predicting..."):
                processed_batch = preprocess_input(batch_df)
                preds = model.predict(processed_batch)
                probs = model.predict_proba(processed_batch)[:, 1]
            
            results_df = batch_df.copy()
            results_df['Prediction'] = ['>50K' if p==1 else '<=50K' for p in preds]
            results_df['Probability_>50K'] = probs.round(3)
            
            st.markdown("---")
            st.subheader("Results for Uploaded Data")
            st.dataframe(results_df)

            csv_buffer = io.StringIO()
            results_df.to_csv(csv_buffer, index=False)
            st.download_button(label="⬇️ Download Predictions as CSV",
                                data=csv_buffer.getvalue(),
                                file_name="salary_predictions.csv",
                                mime="text/csv")
            
        except Exception as e:
            st.error(f"❌ Failed to process uploaded file: {e}")

st.sidebar.markdown(f"---")
st.sidebar.info(f"**Current date:** {datetime.now().strftime('%A, %B %d, %Y, %I:%M %p %Z')}")