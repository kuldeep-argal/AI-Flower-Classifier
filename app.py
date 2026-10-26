import streamlit as st
import pandas as pd
import joblib

# Load artifact
artifact = joblib.load("wine_classifier.pkl")
model = artifact["model"]
classes = artifact["classes"]

st.set_page_config(page_title="Wine Cultivar Predictor", page_icon="🍷", layout="centered")

st.title("🍷 Wine Cultivar Classifier")
st.caption("Traditional ML (Random Forest) running on scikit-learn's built-in Wine dataset.")

st.markdown("### Enter Chemical Profile")

col1, col2 = st.columns(2)

with col1:
    alcohol = st.slider("Alcohol (% by volume)", min_value=11.0, max_value=15.0, value=13.0, step=0.1)
    malic_acid = st.slider("Malic Acid (g/L)", min_value=0.5, max_value=6.0, value=2.3, step=0.1)
    color_intensity = st.slider("Color Intensity", min_value=1.0, max_value=13.0, value=5.0, step=0.1)

with col2:
    flavanoids = st.slider("Flavanoids", min_value=0.3, max_value=5.0, value=2.0, step=0.1)
    proline = st.slider("Proline (mg/L)", min_value=270, max_value=1700, value=750, step=10)

input_df = pd.DataFrame([{
    "alcohol": alcohol,
    "malic_acid": malic_acid,
    "color_intensity": color_intensity,
    "flavanoids": flavanoids,
    "proline": proline
}])

if st.button("Classify Wine", type="primary"):
    prediction_idx = model.predict(input_df)[0]
    probabilities = model.predict_proba(input_df)[0]
    predicted_class = classes[prediction_idx].capitalize()

    st.success(f"Predicted Cultivar: **{predicted_class}**")

    # Display probabilities for each class
    prob_df = pd.DataFrame({
        "Cultivar": [c.capitalize() for c in classes],
        "Confidence": [f"{p * 100:.1f}%" for p in probabilities]
    })
    st.table(prob_df)