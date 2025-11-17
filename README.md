🏠 House Price Prediction App
Overview

This is a Streamlit-based web application that predicts house prices using XGBoost Regression. The app is designed to be interactive, user-friendly, and production-ready. Users can input house features and instantly get a predicted price. The app also provides visualization, prediction history, and dashboards for deeper insights.

Features

Predict house prices in real-time based on user inputs

Interactive dashboard with filters for predicted prices, rooms, and income

Visualization of Predicted vs Actual Prices

Live map showing locations of predicted houses

Save and view prediction history, with CSV download option

Dark mode interface with modern styling

Pre-trained model stored to avoid retraining

Fully responsive and deployed on Streamlit Cloud

Project Structure
house-price-prediction-app/
│ app.py                     # Main Streamlit application
│ requirements.txt           # Dependencies
│ model/
│   └── house_model.pkl      # Saved XGBoost model
│ assets/
│   └── pexels-pixabay-280229.jpg   # Sidebar image
│ history.csv                # Stores user predictions (generated at runtime)

Dataset

The app uses the California Housing dataset from sklearn.datasets.fetch_california_housing.

Features used for prediction:

Median Income (MedInc)

House Age (HouseAge)

Average Rooms (AveRooms)

Average Bedrooms (AveBedrms)

Population (Population)

Average Occupancy (AveOccup)

Latitude (Latitude)

Longitude (Longitude)

Installation

Clone the repository

git clone https://github.com/bobby138-web/house-price-prediction-app.git
cd house-price-prediction-app


Install dependencies

pip install -r requirements.txt


Run the app locally

streamlit run app.py

Deployment

The app is fully deployed on Streamlit Cloud.
To deploy your own version:

Push your repo to GitHub

Go to Streamlit Cloud

Select your repository

Set main file as app.py

Click Deploy

Usage

Open the app in a browser

Navigate to 🏡 Predict Price from the sidebar

Enter house details (Median Income, House Age, Rooms, Bedrooms, Population, Occupancy, Latitude, Longitude)

Click Predict Price

View predictions on map, history, or dashboard

Libraries & Dependencies

Streamlit
 – Web app framework

XGBoost
 – Regression model

Pandas
 – Data handling

NumPy
 – Numerical operations

Plotly
 – Interactive charts

Scikit-learn
 – Dataset and metrics

Joblib
 – Model serialization

Screenshots

(Add screenshots of your app running, map view, dashboard, or predicted vs actual plots here)

Author

Fredric Bobby

GitHub: https://github.com/bobby138-web

License

This project is open source and available under the MIT License.
