# app1_improved.py
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from churn_model import ChurnModel
from typing import Dict, Any, Tuple, Optional

# --- 1. PAGE CONFIGURATION ---
# Set the page configuration. This must be the first Streamlit command.
st.set_page_config(
    page_title="Customer Churn Predictor",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- 2. MODEL LOADING ---
@st.cache_resource
def load_model() -> Optional[ChurnModel]:
    """
    Loads and caches the churn prediction model.
    Returns the model object or None if loading fails.
    """
    try:
        churn_model = ChurnModel()
        churn_model.load_model()
        return churn_model
    except FileNotFoundError:
        st.error("❌ Model file not found. Please ensure `churn_model.pkl` is in the correct directory.")
        return None
    except Exception as e:
        st.error(f"❌ Failed to load model. An unexpected error occurred: {e}")
        return None

# --- 3. UI HELPER FUNCTIONS ---
def display_user_inputs() -> Dict[str, Any]:
    """
    Creates and displays the input widgets for user data in the sidebar.
    Returns a dictionary of the collected inputs.
    """
    st.sidebar.header("👤 Customer Details")
    
    # Use the sidebar for inputs to keep the main area clean for results.
    tenure = st.sidebar.number_input("Tenure (months)", 0, 100, 12, help="How many months the customer has been with the company.")
    monthly_charges = st.sidebar.number_input("Monthly Charges ($)", 0.0, 500.0, 50.0, help="The customer's current monthly bill.")
    
    # Automatically calculate TotalCharges but allow user override.
    # The float conversion is important for correctness.
    total_charges = st.sidebar.number_input("Total Charges ($)", 0.0, 10000.0, float(tenure * monthly_charges))
    
    contract = st.sidebar.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])
    payment_method = st.sidebar.selectbox("Payment Method", ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"])
    online_security = st.sidebar.selectbox("Online Security", ["No", "Yes", "No internet service"])
    tech_support = st.sidebar.selectbox("Tech Support", ["No", "Yes", "No internet service"])
    
    with st.sidebar.expander("Additional Services"):
        internet_service = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
        online_backup = st.selectbox("Online Backup", ["No", "Yes", "No internet service"])
        paperless_billing = st.selectbox("Paperless Billing", ["No", "Yes"])

    input_data = {
        "tenure": tenure,
        "MonthlyCharges": monthly_charges,
        "TotalCharges": total_charges,
        "Contract": contract,
        "PaymentMethod": payment_method,
        "OnlineSecurity": online_security,
        "TechSupport": tech_support,
        "InternetService": internet_service,
        "OnlineBackup": online_backup,
        "PaperlessBilling": paperless_billing
    }
    return input_data

def display_prediction_results(prediction: int, proba: Tuple[float, float]):
    """
    Displays the prediction result, confidence, and a detailed explanation.
    
    Args:
        prediction (int): The binary prediction from the model (0 or 1).
        proba (Tuple[float, float]): The prediction probabilities for class 0 and 1.
    """
    st.header("📈 Prediction Result")
    
    churn_status = "Churn Risk" if prediction == 1 else "Likely to Stay"
    confidence = proba[1] if prediction == 1 else proba[0]
    
    # Use st.metric for a clean, modern display of the result.
    delta_color = "inverse" if prediction == 0 else "normal"
    st.metric(label="Prediction", value=churn_status, delta=f"Confidence: {confidence:.2%}", delta_color=delta_color)

    # Use st.progress for a more intuitive visual confidence bar.
    st.write("Confidence Breakdown:")
    st.progress(int(confidence * 100), text=f"{confidence:.0%} Certainty")
    
    # Use color-coded messages for impact and clarity.
    if prediction == 1:
        st.error(f"The model predicts this customer is at a **high risk of churning** with {confidence:.2%} confidence.", icon="🚨")
    else:
        st.success(f"The model predicts this customer will **likely stay** with {confidence:.2%} confidence.", icon="✅")

    with st.expander("🔍 What does this mean?"):
        if prediction == 1:
            st.markdown("""
            - **Churn Risk**: Indicates the customer is likely to cancel their subscription.
            - **Confidence**: Reflects how sure the model is. A high value suggests strong indicators for churn have been detected (e.g., short tenure, high monthly charges, month-to-month contract).
            - **Recommendation**: Proactive engagement is recommended. Consider offering incentives or support.
            """)
        else:
            st.markdown("""
            - **Likely to Stay**: Indicates the customer is predicted to continue using the service.
            - **Confidence**: Shows the model's certainty. A lower confidence (e.g., 55-70%) means the customer may have some underlying risk factors that are worth monitoring, even if they are not predicted to churn immediately.
            - **Recommendation**: Continue providing good service. No immediate action is required based on this prediction.
            """)

def display_feature_importance(model: ChurnModel):
    """
    Calculates and displays feature importance as a horizontal bar chart and a DataFrame.
    
    Args:
        model (ChurnModel): The trained model object.
    """
    st.header("🔍 Key Factors Driving Predictions")
    st.write("This chart shows which customer attributes most influence the model's predictions.")
    
    # Check if the model and its necessary attributes are available.
    if model and hasattr(model, 'model') and hasattr(model.model, 'feature_importances_'):
        feature_names = getattr(model.model, 'feature_names_in_', getattr(model, 'feature_names', None))
        
        if feature_names is None:
            st.warning("Feature names are not available in the model object.")
            return

        imp_df = pd.DataFrame({
            'Feature': feature_names,
            'Importance': model.model.feature_importances_
        }).sort_values(by='Importance', ascending=True)

        # Create a more readable horizontal bar chart.
        fig, ax = plt.subplots(figsize=(12, 8))
        ax.barh(imp_df['Feature'], imp_df['Importance'], color='skyblue')
        ax.set_xlabel('Importance')
        ax.set_title('Feature Importance')
        plt.tight_layout()
        st.pyplot(fig)

        st.write("The table below ranks the features by their influence on the churn prediction.")
        st.dataframe(imp_df.sort_values(by='Importance', ascending=False), use_container_width=True)
    else:
        st.warning("Feature importance information is not available for this model type.")

# --- 4. MAIN APPLICATION ---
def main():
    """Main function to orchestrate the Streamlit app."""
    
    st.title("📊 Customer Churn Prediction Dashboard")
    st.write("This interactive dashboard uses a machine learning model to predict customer churn based on their details and service usage.")
    
    model = load_model()
    if model is None:
        st.warning("Please resolve the model loading issue to proceed.")
        st.stop() # Stop execution if the model could not be loaded.

    # Collect user inputs from the sidebar.
    input_data = display_user_inputs()
    
    # Use a primary button in the sidebar for the main action.
    if st.sidebar.button("🚀 Predict Churn", type="primary", use_container_width=True):
        try:
            # The main content area is now used for results.
            with st.spinner('Analyzing customer data...'):
                prediction, proba = model.predict(input_data)
                display_prediction_results(prediction, proba)
                st.divider()
                display_feature_importance(model)
        except Exception as e:
            st.error(f"An error occurred during prediction: {e}")

    # Add an 'About' section at the bottom of the sidebar.
    st.sidebar.divider()
    st.sidebar.header("About This App")
    st.sidebar.info(
        """
        This app was created by **Tushar Chaudhari**.
        
        **Technologies:** Python, Streamlit, Pandas, Scikit-learn, XGBoost.
        
        [View Source Code on GitHub](https://github.com/iTushar09/Customers-Churn-Prediction-App)
        """
    )

 

if __name__ == "__main__":
    main()