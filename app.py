import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image

# Load the saved model
model = joblib.load('credit_risk_model.pkl')

# Set page config
st.set_page_config(
    page_title="Credit Risk Prediction",
    page_icon=":money_with_wings:",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Sidebar
with st.sidebar:
    st.title("💳 Credit Risk Predictor")
    st.image("https://cdn-icons-png.flaticon.com/512/477/477103.png", width=100)
    st.markdown("""
    Predict the credit risk of loan applicants using machine learning.
    
    ### How to Use:
    - Enter a single applicant's details
    - Upload a CSV for batch prediction
    - Click **Predict** to get results
    """)
    st.markdown("Built with ❤️ for the Hackathon")

# Main content
st.title("🏦 Credit Risk Predictor")
st.markdown("Determine whether a loan applicant is at **Low** or **High** credit risk.")

# Create tabs
tab1, tab2, tab3 = st.tabs(["📍 Single Prediction", "📁 Batch Prediction", "📊 Data Insights"])

with tab1:
    st.header("🔍 Single Applicant Prediction")
    
    with st.form("single_pred_form"):
        st.write("### Enter Applicant Information:")
        col1, col2 = st.columns(2)

        with col1:
            age = st.number_input("Age", min_value=18, max_value=100, value=30)
            sex = st.selectbox("Sex", ["male", "female"])
            job = st.selectbox("Job", [0, 1, 2, 3], 
                             help="0: unskilled, 1: skilled, 2: highly skilled, 3: executive")
            housing = st.selectbox("Housing", ["own", "rent", "free"])

        with col2:
            saving_accounts = st.selectbox(
                "Saving Accounts", 
                ["none", "little", "moderate", "quite rich", "rich"]
            )
            checking_account = st.selectbox(
                "Checking Account", 
                ["none", "little", "moderate", "rich"]
            )
            duration = st.number_input("Loan Duration (months)", min_value=1, max_value=100, value=12)
            purpose = st.selectbox(
                "Purpose of Loan",
                ["radio/TV", "education", "furniture/equipment", "car", 
                 "business", "domestic appliances", "repairs", "vacation/others"]
            )

        submitted = st.form_submit_button("🚀 Predict Credit Risk")

        if submitted:
            input_data = pd.DataFrame({
                'Age': [age],
                'Sex': [sex],
                'Job': [job],
                'Housing': [housing],
                'Saving accounts': [saving_accounts],
                'Checking account': [checking_account],
                'Duration': [duration],
                'Purpose': [purpose]
            })

            prediction = model.predict(input_data)
            proba = model.predict_proba(input_data)

            st.subheader("🧾 Prediction Result")
            if prediction[0] == 0:
                st.success("✅ Low Credit Risk")
                st.balloons()
            else:
                st.error("❌ High Credit Risk")
                st.subheader("Recommendations to reduce the risk")
                st.markdown("""
- **Prefer shorter loan durations** to reduce risk.
- **Support young applicants** with financial guidance.
- **Encourage opening bank accounts** for transparency.
- **Promote homeownership** over renting.
- **Separate business and personal loans** for clarity.
- **Invest in skill development** for unskilled workers.
- **Combine model predictions with expert judgment**.
""")


            st.metric("Confidence Level", f"{max(proba[0])*100:.1f}%")

            fig, ax = plt.subplots()
            ax.barh(['Low Risk', 'High Risk'], proba[0], color=['green', 'red'])
            ax.set_xlim(0, 1)
            ax.set_title('Prediction Probability')
            st.pyplot(fig)
            

with tab2:
    st.header("📂 Batch Prediction")
    st.markdown("Upload a CSV file with multiple applicants to get predictions.")

    uploaded_file = st.file_uploader("Upload CSV", type="csv")

    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            st.write("📄 Preview of Uploaded Data:")
            st.dataframe(df.head())

            if st.button("Predict for All Applicants"):
                with st.spinner('🔄 Running predictions...'):
                    predictions = model.predict(df)
                    probas = model.predict_proba(df)

                    results = df.copy()
                    results['Prediction'] = predictions
                    results['Prediction'] = results['Prediction'].map({0: 'Low', 1: 'High'})
                    results['Confidence'] = np.max(probas, axis=1)

                    st.success("✅ Batch Predictions Complete")
                    st.dataframe(results)

                    csv = results.to_csv(index=False).encode('utf-8')
                    st.download_button(
                        "📥 Download Predictions",
                        csv,
                        "credit_predictions.csv",
                        "text/csv",
                        key='download-csv'
                    )

                    st.subheader("📊 Prediction Summary")
                    fig, ax = plt.subplots()
                    results['Prediction'].value_counts().plot(kind='bar', ax=ax, color=['green', 'red'])
                    ax.set_title('Low vs High Credit Risk Counts')
                    st.pyplot(fig)

        except Exception as e:
            st.error(f"❗ Error processing file: {str(e)}")

with tab3:
    st.header("📈 Data Insights")
    st.markdown("Explore insights about the dataset and the model.")

    st.subheader("🔑 Key Influential Features")
    features = [
        "Duration", "Age", "Checking Account (rich)", "Checking Account (moderate)",
        "Saving Accounts (rich)", "Housing (rent)", "Purpose (business)",
        "Checking Account (little)", "Saving Accounts (quite rich)", "Job"
    ]
    importance = [0.25, 0.18, 0.07, 0.06, 0.05, 0.05, 0.04, 0.04, 0.03, 0.03]

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(features[::-1], importance[::-1], color='skyblue')
    ax.set_title('Feature Importance')
    st.pyplot(fig)

    st.subheader("📊 Credit Risk Distribution in Dataset")
    fig, ax = plt.subplots()
    ax.pie([500, 500], labels=['Low Risk', 'High Risk'], autopct='%1.1f%%',
           colors=['lightgreen', 'lightcoral'])
    ax.set_title('Credit Risk Balance')
    st.pyplot(fig)

    st.subheader("📏 Model Performance")
    metrics = pd.DataFrame({
        'Metric': ['Accuracy', 'Precision', 'Recall', 'F1 Score', 'ROC AUC'],
        'Value': [0.82, 0.81, 0.83, 0.82, 0.89]
    })
    st.dataframe(metrics.style.highlight_max(axis=0))

    st.subheader("📋 Evaluation Recommendations")
    st.markdown("""
    - **Shorter Loan Durations** generally indicate lower risk.
    - **Younger Applicants** may present higher risk.
    - **No Bank Accounts** often correlate with higher risk.
    - **Renters** typically have higher risk than owners.
    - **Business Loans** tend to be riskier than personal use.
    - **Unskilled Jobs** are associated with higher credit risk.
    - Use model predictions **in conjunction with human judgment**.
    """)

