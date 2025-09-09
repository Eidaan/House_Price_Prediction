import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import shap
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
import time
import os

# -------------------------------
# Create necessary folders if not exist
# -------------------------------
os.makedirs("models", exist_ok=True)
os.makedirs("data", exist_ok=True)

# -------------------------------
# Page config
# -------------------------------
st.set_page_config(page_title="🏠 House Price Dashboard", layout="wide")

# -------------------------------
# Solo-Leveling Dark Theme
# -------------------------------
st.markdown("""
<style>
.stApp {background-color: #0D0D0D; color: #FFFFFF;}
.stButton>button {background-color: #FFD700; color: #000000; font-weight: bold;}
.stSidebar {background-color: #1A1A1A; color: #FFFFFF;}
h1,h2,h3 {color: #FFD700;}
</style>
""", unsafe_allow_html=True)

# -------------------------------
# Load Dataset
# -------------------------------
@st.cache_data
def load_data():
    data = fetch_california_housing()
    X = pd.DataFrame(data.data, columns=data.feature_names)
    y = pd.Series(data.target, name="Target")
    # Feature engineering
    X["RoomsPerHousehold"] = X["AveRooms"] / X["AveOccup"]
    X["BedroomsPerRoom"] = X["AveBedrms"] / X["AveRooms"]
    X["PopulationPerHousehold"] = X["Population"] / X["AveOccup"]
    return X, y

X, y = load_data()

# -------------------------------
# Train Models
# -------------------------------
def train_models():
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Linear Regression
    lr = LinearRegression()
    lr.fit(X_train, y_train)
    
    # Random Forest
    rf = RandomForestRegressor(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)
    
    # Gradient Boosting
    gb = GradientBoostingRegressor(n_estimators=200, learning_rate=0.1, random_state=42)
    gb.fit(X_train, y_train)
    
    # Save models to "models/" folder
    joblib.dump(lr, "models/linear_model_exp.joblib")
    joblib.dump(rf, "models/rf_model_exp.joblib")
    joblib.dump(gb, "models/gb_model_exp.joblib")
    
    # Metrics
    metrics = {}
    for name, model in [("Linear Regression", lr), ("Random Forest", rf), ("Gradient Boosting", gb)]:
        y_pred = model.predict(X_test)
        metrics[name] = {
            "MSE": round(mean_squared_error(y_test, y_pred), 4),
            "R2": round(r2_score(y_test, y_pred), 4),
            "MAE": round(mean_absolute_error(y_test, y_pred), 4)
        }
    return metrics

try:
    lr_model = joblib.load("models/linear_model_exp.joblib")
    rf_model = joblib.load("models/rf_model_exp.joblib")
    gb_model = joblib.load("models/gb_model_exp.joblib")
    metrics = None
except:
    st.info("Training models, please wait...")
    metrics = train_models()
    lr_model = joblib.load("models/linear_model_exp.joblib")
    rf_model = joblib.load("models/rf_model_exp.joblib")
    gb_model = joblib.load("models/gb_model_exp.joblib")
    st.success("Models trained successfully!")

# -------------------------------
# Sidebar: Model Selection
# -------------------------------
st.sidebar.title("⚙️ Settings")
model_choice = st.sidebar.selectbox("Choose Model", ["Linear Regression", "Random Forest", "Gradient Boosting"])
model = {"Linear Regression": lr_model, "Random Forest": rf_model, "Gradient Boosting": gb_model}[model_choice]

# -------------------------------
# Tabs
# -------------------------------
tabs = st.tabs([
    "📖 Welcome",
    "🏠 Single Prediction",
    "📁 Batch Prediction",
    "📊 Insights & Maps"
])

# -------------------------------
# Tab 0: Welcome
# -------------------------------
with tabs[0]:
    st.header("Welcome to the House Price Prediction Dashboard!")
    st.markdown("""
    This app predicts California house prices using multiple ML models with interactive visualizations.
    Use the tabs above to explore:
    - Single Prediction
    - Batch Prediction
    - Insights & Maps
    """)

# -------------------------------
# Single Prediction Tab
# -------------------------------
with tabs[1]:
    with st.expander("📖 How to Use Single Prediction Tab", expanded=True):
        st.markdown("""
        - Adjust sliders for each feature: Median Income, House Age, Rooms, Bedrooms, Population, Avg Occupancy, Latitude, Longitude
        - Click **Predict** to see the predicted house price
        - SHAP chart explains **which features increased or decreased** the prediction
        - Experiment with sliders to see interactive effect
        """)

    st.header("Single House Prediction")

    # Realistic ranges (1st to 99th percentile)
    ranges = {
        "MedInc": (X.MedInc.quantile(0.01), X.MedInc.quantile(0.99)),
        "HouseAge": (X.HouseAge.quantile(0.01), X.HouseAge.quantile(0.99)),
        "AveRooms": (X.AveRooms.quantile(0.01), X.AveRooms.quantile(0.99)),
        "AveBedrms": (X.AveBedrms.quantile(0.01), X.AveBedrms.quantile(0.99)),
        "Population": (X.Population.quantile(0.01), X.Population.quantile(0.99)),
        "AveOccup": (X.AveOccup.quantile(0.01), X.AveOccup.quantile(0.99)),
        "Latitude": (X.Latitude.min(), X.Latitude.max()),
        "Longitude": (X.Longitude.min(), X.Longitude.max())
    }

    col1, col2, col3 = st.columns(3)

    # Helper function: validated number input with tooltip & warning
    def validated_input(name, slider_val, min_val, max_val):
        st.markdown(f"<span title='Realistic range: {round(min_val,2)} - {round(max_val,2)}'>{name}</span>", unsafe_allow_html=True)
        val = st.number_input(
            f"Type {name}",
            value=float(slider_val),
            min_value=float(min_val),
            max_value=float(max_val),
            format="%.4f"
        )
        if val < min_val or val > max_val:
            st.warning(f"⚠️ {name} should be between {round(min_val,2)} and {round(max_val,2)}")
        return val

    with col1:
        MedInc = st.slider("Median Income (10k $)", ranges["MedInc"][0], ranges["MedInc"][1], float(X.MedInc.mean()))
        MedInc_input = validated_input("Median Income (10k $)", MedInc, *ranges["MedInc"])

        HouseAge = st.slider("House Age", ranges["HouseAge"][0], ranges["HouseAge"][1], float(X.HouseAge.mean()))
        HouseAge_input = validated_input("House Age", HouseAge, *ranges["HouseAge"])

        AveRooms = st.slider("Average Rooms", ranges["AveRooms"][0], ranges["AveRooms"][1], float(X.AveRooms.mean()))
        AveRooms_input = validated_input("Average Rooms", AveRooms, *ranges["AveRooms"])

    with col2:
        AveBedrms = st.slider("Average Bedrooms", ranges["AveBedrms"][0], ranges["AveBedrms"][1], float(X.AveBedrms.mean()))
        AveBedrms_input = validated_input("Average Bedrooms", AveBedrms, *ranges["AveBedrms"])

        Population = st.slider("Population", ranges["Population"][0], ranges["Population"][1], float(X.Population.mean()))
        Population_input = validated_input("Population", Population, *ranges["Population"])

        AveOccup = st.slider("Average Occupancy", ranges["AveOccup"][0], ranges["AveOccup"][1], float(X.AveOccup.mean()))
        AveOccup_input = validated_input("Average Occupancy", AveOccup, *ranges["AveOccup"])

    with col3:
        Latitude = st.slider("Latitude", ranges["Latitude"][0], ranges["Latitude"][1], float(X.Latitude.mean()))
        Latitude_input = validated_input("Latitude", Latitude, *ranges["Latitude"])

        Longitude = st.slider("Longitude", ranges["Longitude"][0], ranges["Longitude"][1], float(X.Longitude.mean()))
        Longitude_input = validated_input("Longitude", Longitude, *ranges["Longitude"])

    # Input dataframe
    input_df = pd.DataFrame({
        'MedInc':[MedInc_input],
        'HouseAge':[HouseAge_input],
        'AveRooms':[AveRooms_input],
        'AveBedrms':[AveBedrms_input],
        'Population':[Population_input],
        'AveOccup':[AveOccup_input],
        'Latitude':[Latitude_input],
        'Longitude':[Longitude_input],
        'RoomsPerHousehold':[AveRooms_input/AveOccup_input],
        'BedroomsPerRoom':[AveBedrms_input/AveRooms_input],
        'PopulationPerHousehold':[Population_input/AveOccup_input]
    })

    if st.button("Predict"):
        prediction = model.predict(input_df)[0]
        st.markdown(f"""
        <div style='background-color:#1A1A1A; border:2px solid #FFD700; padding:15px; border-radius:15px; text-align:center;'>
            <h2>🏠 Predicted House Price</h2>
            <h1 style='color:#FFD700;'>${round(prediction*100000,2)}</h1>
        </div>
        """, unsafe_allow_html=True)

        explainer = shap.Explainer(model, X)
        shap_values = explainer(input_df)
        st.subheader("Feature Contributions (SHAP)")
        shap.plots.bar(shap_values, show=False)
        st.pyplot(plt.gcf())
        plt.clf()


# -------------------------------
# Tab 2: Batch Prediction
# -------------------------------
with tabs[2]:
    with st.expander("📖 How to Use Batch Prediction Tab", expanded=True):
        st.markdown("""
        - Upload a CSV file with all required features
        - Click **Predict** to get predictions for all rows
        - View table with predicted prices
        - Download results using **Download Predictions** button
        - Top 10% table shows most expensive houses
        - Progress bar shows real-time feedback during prediction
        """)
    st.header("Batch Prediction")
    
    uploaded_file = st.file_uploader("Upload CSV with house features", type=["csv"], key="batch")
    if uploaded_file is not None:
        batch_df = pd.read_csv(uploaded_file)

        # Clip inputs to realistic ranges
        for col, (min_val, max_val) in ranges.items():
            if col in batch_df.columns:
                batch_df[col] = batch_df[col].clip(min_val, max_val)

        st.subheader("Processing Predictions...")
        progress = st.progress(0)
        predictions = []
        for i in range(len(batch_df)):
            pred = model.predict(batch_df.iloc[[i]])[0]
            predictions.append(pred*100000)
            progress.progress(int((i+1)/len(batch_df)*100))
            time.sleep(0.01)
        batch_df["PredictedPrice"] = predictions
        st.write(batch_df)
        
        batch_df.to_csv("data/batch_predictions.csv", index=False)
        st.download_button("Download Predictions", "data/batch_predictions.csv", "predictions.csv")
        
        st.subheader("📊 Batch Statistics")
        st.write(batch_df["PredictedPrice"].describe())
        st.subheader("🏆 Top 10% Most Expensive Houses")
        top10 = batch_df[batch_df["PredictedPrice"] >= batch_df["PredictedPrice"].quantile(0.9)]
        st.dataframe(top10)

# -------------------------------
# Tab 3: Insights & Maps
# -------------------------------
with tabs[3]:
    with st.expander("📖 How to Use Insights & Maps Tab", expanded=True):
        st.markdown("""
        - Feature Importance shows which variables drive predictions
        - Correlation Heatmap shows relationships between features and target
        - Map displays predicted prices geographically (color = price, size = magnitude)
        - Model Metrics show MSE, R², and MAE for selected model
        """)
    st.header("EDA & Model Insights")
    
    st.subheader("Feature Importance (Random Forest)")
    sns.set_style("darkgrid")
    plt.figure(figsize=(10,5))
    importances = pd.Series(rf_model.feature_importances_, index=X.columns)
    importances.sort_values().plot(kind="barh", color="#FFD700")
    st.pyplot(plt.gcf())
    plt.clf()
    
    st.subheader("Correlation Heatmap")
    plt.figure(figsize=(10,8))
    sns.heatmap(X.corr(), annot=True, cmap="viridis")
    st.pyplot(plt.gcf())
    plt.clf()
    
    st.subheader("Predicted Prices Map")
    map_df = X.copy()
    map_df["PredictedPrice"] = rf_model.predict(X) * 100000
    map_df["PredictedPrice"] = np.clip(map_df["PredictedPrice"], 0, None)
    map_df["MarkerSize"] = map_df["PredictedPrice"].apply(lambda x: max(x/100000, 5))
    fig = px.scatter_mapbox(
        map_df,
        lat="Latitude",
        lon="Longitude",
        color="PredictedPrice",
        size="MarkerSize",
        color_continuous_scale="Viridis",
        zoom=5,
        height=500
    )
    fig.update_layout(mapbox_style="carto-darkmatter")
    st.plotly_chart(fig)
    
    if metrics:
        st.subheader("Model Performance Metrics")
        st.write(metrics)
