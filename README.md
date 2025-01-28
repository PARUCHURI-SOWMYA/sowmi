import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
import matplotlib.pyplot as plt
import plotly.express as px
import datetime

# Title
st.title("🛡️ Dark Tracer: Early Detection Framework for Malware Activity")
st.write("Detect anomalous spatiotemporal patterns in your data.")

# Sidebar for file upload
st.sidebar.header("Upload Data")
uploaded_file = st.sidebar.file_uploader("Upload a CSV file with spatiotemporal data", type=["csv"])

# Input parameters
st.sidebar.header("Anomaly Detection Parameters")
contamination = st.sidebar.slider("Contamination (Anomaly Fraction)", 0.01, 0.5, 0.1)
n_estimators = st.sidebar.slider("Number of Estimators", 50, 500, 100)

# Load and display data
if uploaded_file:
    # Read the uploaded file
    data = pd.read_csv(uploaded_file)
    st.write("### Uploaded Data")
    st.dataframe(data.head())

    # Expecting columns: timestamp, location (latitude/longitude), and activity
    if 'timestamp' in data.columns and 'latitude' in data.columns and 'longitude' in data.columns:
        # Parse timestamp if needed
        if data['timestamp'].dtype != 'datetime64[ns]':
            data['timestamp'] = pd.to_datetime(data['timestamp'])

        # Plot data distribution
        st.write("### Spatiotemporal Data Distribution")
        fig = px.scatter_mapbox(
            data,
            lat="latitude",
            lon="longitude",
            color="activity",
            hover_data=["timestamp"],
            title="Activity Distribution",
            mapbox_style="open-street-map",
            height=500
        )
        st.plotly_chart(fig)

        # Feature extraction for anomaly detection
        st.write("### Anomaly Detection")
        data['hour'] = data['timestamp'].dt.hour
        features = data[['latitude', 'longitude', 'hour']]

        # Anomaly detection using Isolation Forest
        model = IsolationForest(n_estimators=n_estimators, contamination=contamination, random_state=42)
        data['anomaly'] = model.fit_predict(features)
        data['anomaly'] = data['anomaly'].apply(lambda x: 'Anomalous' if x == -1 else 'Normal')

        # Display anomaly counts
        st.write("### Anomaly Counts")
        st.bar_chart(data['anomaly'].value_counts())

        # Plot anomalies on map
        st.write("### Detected Anomalies")
        anomalies = data[data['anomaly'] == 'Anomalous']
        fig_anomalies = px.scatter_mapbox(
            anomalies,
            lat="latitude",
            lon="longitude",
            color="anomaly",
            hover_data=["timestamp"],
            title="Anomalous Activities",
            mapbox_style="open-street-map",
            height=500
        )
        st.plotly_chart(fig_anomalies)

        # Save processed data
        st.sidebar.download_button(
            "Download Processed Data",
            data.to_csv(index=False).encode('utf-8'),
            "processed_data.csv",
            "text/csv"
        )

    else:
        st.error("Uploaded data must contain 'timestamp', 'latitude', and 'longitude' columns.")
else:
    st.info("Please upload a dataset to begin.")

# Security Best Practices
st.sidebar.header("Best Practices")
st.sidebar.write("""
- Ensure data contains accurate timestamps and geolocations.
- Monitor anomalies in real time for better detection.
- Validate and sanitize inputs to prevent vulnerabilities.
""")
