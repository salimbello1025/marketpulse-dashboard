import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.linear_model import LinearRegression
from streamlit_autorefresh import st_autorefresh
import os

# =================================
# PAGE CONFIGURATION
# =================================
st.set_page_config(page_title="MarketPulse", layout="wide")

# Auto refresh every 60 seconds
st_autorefresh(interval=60000, key="datarefresh")

st.title("MarketPulse: Real-Time Food Price Intelligence")
st.caption("Monitoring food prices across Nigerian markets")

# =================================
# LOAD DATA
# =================================
@st.cache_data
def load_data():

    file_path = "food_prices_nigeria.csv"

    if not os.path.exists(file_path):
        st.error("Dataset 'food_prices_nigeria.csv' not found.")
        st.stop()

    df = pd.read_csv(file_path)

    df["date"] = pd.to_datetime(
        df["date"],
        format="mixed",
        errors="coerce"
    )

    df = df.dropna(subset=["date"])

    return df


data = load_data()

# =================================
# SIDEBAR NAVIGATION
# =================================
menu = st.sidebar.radio(
    "Navigation",
    [
        "Dashboard",
        "Market Map",
        "State Price Map",
        "Price Forecast",
        "Submit Market Price",
        "Download Data"
    ]
)

# =================================
# SIDEBAR FILTERS
# =================================
st.sidebar.header("Filters")

commodity = st.sidebar.selectbox(
    "Select Commodity",
    sorted(data["commodity"].unique())
)

regions = st.sidebar.multiselect(
    "Select Regions",
    sorted(data["admin1"].unique()),
    default=sorted(data["admin1"].unique())[:1]
)

filtered = data[
    (data["commodity"] == commodity) &
    (data["admin1"].isin(regions))
]

if filtered.empty:
    st.warning("No data available for this selection.")
    st.stop()

# =================================
# DASHBOARD
# =================================
if menu == "Dashboard":

    st.header("Market Overview")

    col1, col2, col3 = st.columns(3)

    col1.metric("Average Price", round(filtered["price"].mean(), 2))
    col2.metric("Maximum Price", round(filtered["price"].max(), 2))
    col3.metric("Minimum Price", round(filtered["price"].min(), 2))

    sorted_prices = filtered.sort_values("date")

    latest = sorted_prices.iloc[-1]["price"]
    earliest = sorted_prices.iloc[0]["price"]

    inflation = ((latest - earliest) / earliest) * 100

    st.subheader("Inflation Alert")

    if inflation > 15:
        st.error(f"High inflation detected: {round(inflation,2)}%")
    elif inflation > 5:
        st.warning(f"Moderate inflation: {round(inflation,2)}%")
    else:
        st.success("Food prices are stable")

    fig1 = px.line(
        filtered,
        x="date",
        y="price",
        color="admin1",
        markers=True
    )

    market_price = (
        filtered
        .groupby(["admin1","market"])["price"]
        .mean()
        .reset_index()
    )

    fig2 = px.bar(
        market_price,
        x="market",
        y="price",
        color="admin1",
        barmode="group"
    )

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Food Price Trend")
        st.plotly_chart(fig1, use_container_width=True)

    with col2:
        st.subheader("Market Price Comparison")
        st.plotly_chart(fig2, use_container_width=True)

    st.subheader("Top 10 Most Expensive Markets")

    top_markets = (
        filtered
        .groupby("market")["price"]
        .mean()
        .sort_values(ascending=False)
        .head(10)
        .reset_index()
    )

    st.dataframe(top_markets)

    risk = ((latest - earliest) / earliest) * 100

    st.subheader("Food Crisis Risk Indicator")

    if risk > 20:
        st.error("High food security risk")
    elif risk > 10:
        st.warning("Moderate food price stress")
    else:
        st.success("Food market conditions are stable")

    mean_price = filtered["price"].mean()
    std_price = filtered["price"].std()

    filtered["anomaly_score"] = (filtered["price"] - mean_price) / std_price

    anomalies = filtered[filtered["anomaly_score"].abs() > 2]

    st.subheader("AI Price Anomaly Detection")

    if anomalies.empty:
        st.success("No suspicious price entries detected")
    else:
        st.warning("Suspicious prices detected")
        st.dataframe(anomalies[["market","price","anomaly_score"]])

# =================================
# MARKET MAP
# =================================
elif menu == "Market Map":

    st.header("Market Locations")

    fig_map = px.scatter_mapbox(
        filtered,
        lat="latitude",
        lon="longitude",
        hover_name="market",
        size="price",
        zoom=6
    )

    fig_map.update_layout(mapbox_style="open-street-map")

    st.plotly_chart(fig_map, use_container_width=True)

    st.subheader("Food Price Heatmap")

    heatmap = px.density_mapbox(
        filtered,
        lat="latitude",
        lon="longitude",
        z="price",
        radius=20,
        center=dict(lat=9.0820, lon=8.6753),
        zoom=5,
        mapbox_style="open-street-map"
    )

    st.plotly_chart(heatmap, use_container_width=True)

# =================================
# STATE PRICE MAP (FINAL UPGRADE)
# =================================
elif menu == "State Price Map":

    st.header("Nigeria State Food Price Map")

    state_prices = (
        data
        .groupby("admin1")["price"]
        .mean()
        .reset_index()
    )

    fig = px.choropleth(
        state_prices,
        locations="admin1",
        locationmode="country names",
        color="price",
        color_continuous_scale="Reds",
        title="Average Food Prices by State"
    )

    st.plotly_chart(fig, use_container_width=True)

# =================================
# PRICE FORECAST
# =================================
elif menu == "Price Forecast":

    st.header("Price Forecast (Next 30 Days)")

    forecast_data = (
        filtered
        .groupby("date")["price"]
        .mean()
        .reset_index()
    )

    forecast_data["date_ordinal"] = forecast_data["date"].map(pd.Timestamp.toordinal)

    X = forecast_data[["date_ordinal"]]
    y = forecast_data["price"]

    model = LinearRegression()
    model.fit(X,y)

    future_dates = pd.date_range(
        forecast_data["date"].max(),
        periods=30
    )

    future_ordinal = future_dates.map(pd.Timestamp.toordinal)

    predicted_prices = model.predict(
        future_ordinal.values.reshape(-1,1)
    )

    forecast_df = pd.DataFrame({
        "date":future_dates,
        "predicted_price":predicted_prices
    })

    fig_forecast = px.line(
        forecast_df,
        x="date",
        y="predicted_price"
    )

    st.plotly_chart(fig_forecast, use_container_width=True)

# =================================
# SUBMIT MARKET PRICE
# =================================
elif menu == "Submit Market Price":

    st.header("Secure Market Price Submission")

    PASSWORD = "Nigeria001"

    if "auth" not in st.session_state:
        st.session_state.auth = False

    if not st.session_state.auth:

        password = st.text_input(
            "Enter Password",
            type="password"
        )

        if st.button("Login"):

            if password == PASSWORD:
                st.session_state.auth = True
                st.success("Access granted")
            else:
                st.error("Incorrect password")

    if st.session_state.auth:

        with st.form("price_form"):

            market = st.text_input("Market Name")

            region = st.selectbox(
                "State",
                sorted(data["admin1"].unique())
            )

            commodity_name = st.selectbox(
                "Commodity",
                sorted(data["commodity"].unique())
            )

            unit = st.text_input("Unit (kg, mudu, bag)")

            price = st.number_input(
                "Price",
                min_value=0.0
            )

            latitude = st.number_input("Latitude", format="%.6f")
            longitude = st.number_input("Longitude", format="%.6f")

            submit = st.form_submit_button("Submit Price")

            if submit:

                new_entry = pd.DataFrame({
                    "date":[pd.Timestamp.today()],
                    "admin1":[region],
                    "market":[market],
                    "commodity":[commodity_name],
                    "unit":[unit],
                    "price":[price],
                    "latitude":[latitude],
                    "longitude":[longitude]
                })

                updated = pd.concat([data, new_entry], ignore_index=True)

                updated.to_csv("food_prices_nigeria.csv", index=False)

                st.success("Price submitted successfully")

# =================================
# DOWNLOAD DATA
# =================================
elif menu == "Download Data":

    st.header("Secure Data Download")

    PASSWORD = "Nigeria001"

    if "download_auth" not in st.session_state:
        st.session_state.download_auth = False

    if not st.session_state.download_auth:

        password = st.text_input(
            "Enter Password to Download Dataset",
            type="password"
        )

        if st.button("Unlock Download"):

            if password == PASSWORD:
                st.session_state.download_auth = True
                st.success("Access granted")
            else:
                st.error("Incorrect password")

    if st.session_state.download_auth:

        st.download_button(
            label="Download CSV File",
            data=data.to_csv(index=False),
            file_name="food_prices_nigeria.csv",
            mime="text/csv"
        )