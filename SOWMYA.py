import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

# App Title
st.title("🛡️ Dark Tracer: Malware Activity Detection Framework")
st.write("Detect anomalous spatiotemporal patterns in your data.")

# Sidebar for File Upload
uploaded_file = st.sidebar.file_uploader("Upload a CSV file", type=["csv"])

# Sidebar for Parameters
z_score_threshold = st.sidebar.slider("Z-Score Threshold for Anomaly", 2.0, 5.0, 3.0)

if uploaded_file:
    try:
        # Load and Display Data
        data = pd.read_csv(uploaded_file)
        st.write("### Uploaded Data")
        st.dataframe(data.head())

        # Validate Required Columns
        if {'timestamp', 'latitude', 'longitude'}.issubset(data.columns):
            data['timestamp'] = pd.to_datetime(data['timestamp'])  # Convert to datetime
            data['hour'] = data['timestamp'].dt.hour  # Extract hour for anomaly detection

            # Calculate Z-scores for the latitude, longitude, and hour
            data['latitude_zscore'] = (data['latitude'] - data['latitude'].mean()) / data['latitude'].std()
            data['longitude_zscore'] = (data['longitude'] - data['longitude'].mean()) / data['longitude'].std()
            data['hour_zscore'] = (data['hour'] - data['hour'].mean()) / data['hour'].std()

            # Define anomalies based on the Z-score threshold
            data['anomaly'] = np.where(
                (data['latitude_zscore'].abs() > z_score_threshold) |
                (data['longitude_zscore'].abs() > z_score_threshold) |
                (data['hour_zscore'].abs() > z_score_threshold),
                'Anomalous', 'Normal'
            )

            # Display Anomaly Counts
            st.write("### Anomaly Counts")
            st.bar_chart(data['anomaly'].value_counts())

            # Visualize Anomalies on Map
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

            # Download Processed Data
            csv_data = data.to_csv(index=False).encode('utf-8')
            st.sidebar.download_button(
                label="Download Processed Data",
                data=csv_data,
                file_name="processed_data.csv",
                mime="text/csv"
            )
        else:
            st.error("The uploaded file must contain 'timestamp', 'latitude', and 'longitude' columns.")

    except Exception as e:
        st.error(f"Error processing file: {e}")

else:
    st.info("Please upload a dataset to start analyzing.")

# Footer
st.sidebar.info("Powered by Streamlit")
