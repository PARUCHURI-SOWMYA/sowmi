import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
import plotly.express as px

# Title
st.title("🛡️ Dark Tracer: Malware Activity Detection")
st.write("Detect anomalous spatiotemporal patterns in your data.")

# Sidebar for file upload
uploaded_file = st.sidebar.file_uploader("Upload CSV file", type=["csv"])

# Set anomaly detection parameters
contamination = st.sidebar.slider("Anomaly Fraction", 0.01, 0.5, 0.1)
n_estimators = st.sidebar.slider("Estimators", 50, 500, 100)

if uploaded_file:
    try:
        # Load and display data
        data = pd.read_csv(uploaded_file)
        st.write("### Uploaded Data")
        st.dataframe(data.head())

        # Ensure required columns exist
        if {'timestamp', 'latitude', 'longitude'}.issubset(data.columns):
            data['timestamp'] = pd.to_datetime(data['timestamp'])

            # Anomaly detection
            data['hour'] = data['timestamp'].dt.hour
            features = data[['latitude', 'longitude', 'hour']]
            model = IsolationForest(n_estimators=n_estimators, contamination=contamination, random_state=42)
            data['anomaly'] = model.fit_predict(features)
            data['anomaly'] = data['anomaly'].apply(lambda x: 'Anomalous' if x == -1 else 'Normal')

            # Visualize anomalies
            anomalies = data[data['anomaly'] == 'Anomalous']
            fig = px.scatter_mapbox(
                anomalies,
                lat="latitude",
                lon="longitude",
                color="anomaly",
                hover_data=["timestamp"],
                title="Detected Anomalies",
                mapbox_style="open-street-map"
            )
            st.plotly_chart(fig)

            # Export processed data
            csv_data = data.to_csv(index=False).encode('utf-8')
            st.sidebar.download_button("Download Processed Data", csv_data, "processed_data.csv", "text/csv")

        else:
            st.error("Data must include 'timestamp', 'latitude', and 'longitude' columns.")

    except Exception as e:
        st.error(f"Error processing file: {e}")

else:
    st.info("Upload a dataset to begin.")
