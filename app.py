import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import xgboost as xgb
from sklearn.datasets import fetch_california_housing
from PIL import Image
import plotly.express as px
import os

# -----------------------------
# APP CONFIG
# -----------------------------
st.set_page_config(
    page_title="🏠 House Price Prediction",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Dark mode CSS
st.markdown("""
<style>
    body {background-color: #0E1117; color: #FFFFFF;}
    .stButton>button {background-color: #4CAF50; color: white; font-size:16px; border-radius:10px;}
    .stSlider>div>div>div>div {color: #FFFFFF;}
</style>
""", unsafe_allow_html=True)

# -----------------------------
# ASSETS
# -----------------------------
st.sidebar.image("assets/pexels-pixabay-280229.jpg", width=120)
st.sidebar.title("🏠 House Price App")

# -----------------------------
# LOAD OR TRAIN MODEL
# -----------------------------
MODEL_PATH = "model/house_model.pkl"

@st.cache_resource
def load_or_train_model():
    if os.path.exists(MODEL_PATH):
        model = joblib.load(MODEL_PATH)
        return model
    else:
        data = fetch_california_housing(as_frame=True)
        df = data.frame

        X = df.drop("MedHouseVal", axis=1)
        y = df["MedHouseVal"]

        model = xgb.XGBRegressor(
            n_estimators=200,
            learning_rate=0.1,
            max_depth=5,
            random_state=42
        )
        model.fit(X, y)

        os.makedirs("model", exist_ok=True)
        joblib.dump(model, MODEL_PATH)
        return model

model = load_or_train_model()
data_original = fetch_california_housing(as_frame=True).frame
X_orig = data_original.drop("MedHouseVal", axis=1)
y_orig = data_original["MedHouseVal"]

# -----------------------------
# SIDEBAR MENU
# -----------------------------
menu = st.sidebar.radio(
    "Navigation",
    ["🏡 Predict Price", "📊 Dataset Preview", "📈 Prediction History", "🗺 Map View", "📉 Dashboard", "📊 Predicted vs Actual", "ℹ️ About App"]
)

# -----------------------------
# PAGE: Predict Price
# -----------------------------
if menu == "🏡 Predict Price":
    st.title("🏠 House Price Prediction")
    st.write("Enter the house details below:")

    col1, col2 = st.columns(2)
    with col1:
        MedInc = st.number_input("Median Income (10k USD)", 1.0, 20.0, 5.0)
        HouseAge = st.number_input("House Age", 1, 60, 20)
        AveRooms = st.number_input("Average Rooms", 1.0, 10.0, 5.0)
    with col2:
        AveBedrms = st.number_input("Average Bedrooms", 1.0, 5.0, 1.5)
        Population = st.number_input("Population", 1, 50000, 1500)
        Latitude = st.number_input("Latitude", 30.0, 45.0, 35.0)
        Longitude = st.number_input("Longitude", -125.0, -110.0, -120.0)

    user_input = np.array([[MedInc, HouseAge, AveRooms, AveBedrms, Population, Latitude, Longitude]])

    if st.button("🔍 Predict Price"):
        prediction = model.predict(user_input)[0]
        st.success(f"💰 Predicted House Price: **${prediction * 100000:,.2f}**")

        # Save prediction history
        history_file = "history.csv"
        entry = pd.DataFrame([{
            "MedInc": MedInc,
            "HouseAge": HouseAge,
            "AveRooms": AveRooms,
            "AveBedrms": AveBedrms,
            "Population": Population,
            "Latitude": Latitude,
            "Longitude": Longitude,
            "PredictedPrice": prediction * 100000
        }])
        if os.path.exists(history_file):
            entry.to_csv(history_file, mode="a", header=False, index=False)
        else:
            entry.to_csv(history_file, index=False)

# -----------------------------
# PAGE: Dataset Preview
# -----------------------------
elif menu == "📊 Dataset Preview":
    st.header("📊 California Housing Dataset")
    st.dataframe(data_original.head())
    st.write("Dataset Shape:", data_original.shape)

# -----------------------------
# PAGE: Prediction History
# -----------------------------
elif menu == "📈 Prediction History":
    st.header("📈 Prediction History")
    history_file = "history.csv"
    if os.path.exists(history_file):
        df_history = pd.read_csv(history_file)
        st.dataframe(df_history)

        csv = df_history.to_csv(index=False).encode()
        st.download_button("📥 Download CSV", data=csv, file_name="prediction_history.csv", mime="text/csv")

# -----------------------------
# PAGE: Map View
# -----------------------------
elif menu == "🗺 Map View":
    st.header("🗺 Map of Predicted Prices")
    history_file = "history.csv"
    if os.path.exists(history_file):
        df_history = pd.read_csv(history_file)
        st.map(df_history[['Latitude','Longitude']])
        fig = px.scatter_mapbox(
            df_history,
            lat="Latitude",
            lon="Longitude",
            size="PredictedPrice",
            color="PredictedPrice",
            zoom=6,
            mapbox_style="carto-positron",
            hover_data=["PredictedPrice"]
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No predictions yet. Make a prediction first!")

# -----------------------------
# PAGE: Dashboard
# -----------------------------
elif menu == "📉 Dashboard":
    st.header("📉 Interactive Dashboard")
    history_file = "history.csv"
    if os.path.exists(history_file):
        df = pd.read_csv(history_file)
        # Filters
        st.sidebar.subheader("Filters")
        price_range = st.sidebar.slider("Predicted Price Range", float(df.PredictedPrice.min()), float(df.PredictedPrice.max()), (float(df.PredictedPrice.min()), float(df.PredictedPrice.max())))
        room_range = st.sidebar.slider("Number of Rooms", int(df.AveRooms.min()), int(df.AveRooms.max()), (int(df.AveRooms.min()), int(df.AveRooms.max())))
        income_range = st.sidebar.slider("Median Income (10k USD)", float(df.MedInc.min()), float(df.MedInc.max()), (float(df.MedInc.min()), float(df.MedInc.max())))
        filtered = df[
            (df.PredictedPrice >= price_range[0]) & (df.PredictedPrice <= price_range[1]) &
            (df.AveRooms >= room_range[0]) & (df.AveRooms <= room_range[1]) &
            (df.MedInc >= income_range[0]) & (df.MedInc <= income_range[1])
        ]
        st.dataframe(filtered)
        fig1 = px.histogram(filtered, x="PredictedPrice", nbins=30, title="Predicted Price Distribution")
        st.plotly_chart(fig1, use_container_width=True)
        fig2 = px.scatter(filtered, x="AveRooms", y="PredictedPrice", color="MedInc", size="Population", title="Predicted Price vs Rooms")
        st.plotly_chart(fig2, use_container_width=True)
    else:
        st.info("No predictions yet. Make some predictions first!")

# -----------------------------
# PAGE: Predicted vs Actual
# -----------------------------
elif menu == "📊 Predicted vs Actual":
    st.header("📊 Predicted vs Actual Values")
    y_pred = model.predict(X_orig)
    df_plot = pd.DataFrame({"Actual": y_orig * 100000, "Predicted": y_pred * 100000})
    fig = px.scatter(df_plot, x="Actual", y="Predicted", title="Predicted vs Actual Prices", trendline="ols")
    st.plotly_chart(fig, use_container_width=True)

    # Metrics
    mse = mean_squared_error(df_plot.Actual, df_plot.Predicted)
    r2 = r2_score(df_plot.Actual, df_plot.Predicted)
    st.metric("Mean Squared Error", f"{mse:,.2f}")
    st.metric("R² Score", f"{r2:.4f}")

# -----------------------------
# PAGE: About
# -----------------------------
elif menu == "ℹ️ About App":
    st.header("ℹ️ About This App")
    st.write("""
    Production-ready app predicting house prices using **XGBoost Regression**.
    - Interactive dashboard with filters
    - Predicted vs Actual plot
    - MSE & R² metrics
    - Live map of predictions
    - Download prediction history
    - Dark mode & modern UI
    - Saved model (no retraining needed)
    """)
