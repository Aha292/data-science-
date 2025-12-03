import streamlit as st
import pandas as pd
import numpy as np
import geopandas as gpd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression

st.set_page_config(page_title="Punjab Commodity Dashboard", layout="wide")

st.title("Commodity Spot Price Dashboard")

# Set consistent color scheme
BG_COLOR = "#f0f2f6"  # Light gray background similar to Streamlit's default
PLOT_BGCOLOR = "#FFFFFF"  # White plot background
FONT_COLOR = "#000000"  # Changed to black for better visibility

# -----------------------
# Load Potato Data
# -----------------------
df = pd.read_csv(r"c:\Users\abdul\Downloads\amCharts (2).csv")
df = df.drop(['spot_price_2026', 'spot_price_2027'], axis=1)
df["Date"] = pd.to_datetime(df["Date"])

# -----------------------
# Load Other Commodities
# -----------------------
df2 = pd.read_csv(r"c:\Users\abdul\Downloads\amchart,432.csv")
df2 = df2.drop(['spot_price_2026', 'spot_price_2027'], axis=1)
df2["Date"] = pd.to_datetime(df2["Date"])

commodity_list = [
    "Potato", 'Flour', 'Milk',
    "Baspati Ghee-RBD", "Baspati Ghee-Soft Oil 10%", "Baspati Ghee-Soft Oil 20%",
    "Chicken_Broiler", "Daal Chana-Bareek", "Daal Chana-Moti",'DAP','Chaki','basin','Lemon',
    "Onion", "Potato-New", "Potato-Old", "Sugar", "Tomato",'Moong',"Naan","Masoor"
]

# -----------------------
# Commodity Dropdown
# -----------------------
selected_com = st.selectbox("Select a commodity:", commodity_list)

# Create a custom template for consistent styling with improved axis visibility
custom_template = {
    "layout": {
        "paper_bgcolor": PLOT_BGCOLOR,
        "plot_bgcolor": PLOT_BGCOLOR,
        "font": {"color": FONT_COLOR, "size": 12},
        "xaxis": {
            "gridcolor": "#e6e6e6",
            "linecolor": "#e6e6e6",
            "zerolinecolor": "#e6e6e6",
            "title_font": {"size": 14, "color": "black"},
            "tickfont": {"color": "black"},
            "title_standoff": 15  # Added for better spacing
        },
        "yaxis": {
            "gridcolor": "#e6e6e6",
            "linecolor": "#e6e6e6",
            "zerolinecolor": "#e6e6e6",
            "title_font": {"size": 14, "color": "black"},
            "tickfont": {"color": "black"},
            "title_standoff": 15  # Added for better spacing
        }
    }
}
# -----------------------
# Case 1: Potato
# -----------------------
if selected_com == "Potato":
    df_2025_actual = df[df["spot_price_2025"] > 0].copy().reset_index(drop=True)

    # Seasonality avg
    seasonal_prices = []
    for year in ["2020", "2021", "2022", "2023", "2024"]:
        prices = df[f"spot_price_{year}"].values[:52]
        for week, price in enumerate(prices, 1):
            seasonal_prices.append([week, price])
    season_df = pd.DataFrame(seasonal_prices, columns=["Week", "Price"])
    avg_season = season_df.groupby("Week")["Price"].mean()

    # Build data
    all_data = []
    for year in ["2020", "2021", "2022", "2023", "2024"]:
        prices = df[f"spot_price_{year}"].values[:52]
        for week, price in enumerate(prices, 1):
            all_data.append([week, price, year, "Actual"])
    actual_2025 = df["spot_price_2025"][df["spot_price_2025"] > 0].values
    for week, price in enumerate(actual_2025, 1):
        all_data.append([week, price, "2025", "Actual"])
    for week in range(30, 53):
        predicted_price = avg_season[week]
        all_data.append([week, predicted_price, "2025", "Predicted"])
    plot_data = pd.DataFrame(all_data, columns=["Week", "Price", "Year", "Type"])

    # Past years
    fig_full = px.line(
        plot_data[(plot_data["Type"] == "Actual") & (plot_data["Year"] != "2025")],
        x="Week", y="Price", color="Year",
        title="Trend of spot price – Potato (2020–2025)",
        template=custom_template,
        labels={"Week": "Weeks", "Price": "Price"}
    )

    # Full 2025 (one solid black line, thinner now)
    full_2025 = plot_data[plot_data["Year"] == "2025"].sort_values("Week")
    fig_full.add_scatter(
        x=full_2025["Week"], y=full_2025["Price"],
        mode="lines", name="2025 (Actual + Predicted)",
        line=dict(color="black", width=2)   # ✅ reduced width
    )

    # Overlay only predicted part with dotted+markers
    pred_25 = full_2025[full_2025["Type"] == "Predicted"]
    fig_full.add_scatter(
    x=pred_25["Week"], y=pred_25["Price"],
    mode="lines", name="2025 Predicted",
    line=dict(color="red", width=3, dash="dash"),   # 🔹 changed "dot" → "dash"
    marker=dict(size=5, color="red"),
    showlegend=True
)


    # ✅ Highlight predicted area
    fig_full.add_vrect(
        x0=30, x1=52,
        fillcolor="lightgray", opacity=0.3,
        layer="below", line_width=0
    )

    # ✅ Bold labels, ticks, and title
    fig_full.update_layout(
        xaxis_title="Weeks",
        yaxis_title="Price",
        xaxis_title_font=dict(size=14, color="black", family="Arial", weight="bold"),
        yaxis_title_font=dict(size=14, color="black", family="Arial", weight="bold"),
        xaxis=dict(tickfont=dict(size=12, color="black", family="Arial", weight="bold")),
        yaxis=dict(tickfont=dict(size=12, color="black", family="Arial", weight="bold")),
        title_font=dict(size=18, color="black", family="Arial", weight="bold"),
    )

    st.plotly_chart(fig_full, use_container_width=True)

# -----------------------
# Case 2: Other Commodities
# -----------------------
else:
    cdf = df2[df2["commodity_name"] == selected_com].reset_index(drop=True)
    df_2025 = cdf[cdf["spot_price_2025"] > 0].reset_index(drop=True)
    num_weeks = len(df_2025)

    if num_weeks >= 2:
        # --- Train trend ---
        x_train = np.arange(num_weeks).reshape(-1, 1)
        y_train = df_2025["spot_price_2025"]
        model = LinearRegression().fit(x_train, y_train)

        # Predict 20 weeks ahead
        x_pred = np.arange(num_weeks, num_weeks + 20).reshape(-1, 1)
        trend_pred = model.predict(x_pred)

        # Seasonality avg
        seasonal_prices = []
        for yr in ["2020","2021","2022","2023","2024"]:
            if f"spot_price_{yr}" in cdf:
                prices = cdf[f"spot_price_{yr}"].values[:52]
                if len(prices) < 52:
                    prices = np.append(prices, [np.nan]*(52-len(prices)))
                seasonal_prices.extend(prices)
        seasonal_matrix = np.array(seasonal_prices).reshape(5, 52)
        avg_season = np.nanmean(seasonal_matrix, axis=0)

        predicted_prices = []
        for i in range(20):
            wk = num_weeks + i
            season_wk = wk % 52
            combined = (trend_pred[i] + avg_season[season_wk]) / 2
            predicted_prices.append(round(combined, 2))

        # --- Plot ---
        fig = go.Figure()
        colors = ["blue", "red", "green", "purple", "orange"]
        for yr, color in zip(["2020","2021","2022","2023","2024"], colors):
            if f"spot_price_{yr}" in cdf:
                vals = cdf[f"spot_price_{yr}"].values[:52]
                fig.add_trace(go.Scatter(
                    x=list(range(1, len(vals)+1)), y=vals,
                    mode="lines", name=yr,
                    line=dict(color=color), connectgaps=True
                ))

        # Continuous 2025 line (solid, thinner now)
        full_2025_x = list(range(1, num_weeks+len(predicted_prices)+1))
        full_2025_y = list(df_2025["spot_price_2025"]) + predicted_prices
        fig.add_trace(go.Scatter(
            x=full_2025_x, y=full_2025_y,
            mode="lines", name="2025 (Actual + Predicted)",
            line=dict(color="black", width=3)   # ✅ reduced width
        ))

        # Overlay predicted part (dotted + markers)
        fig.add_trace(go.Scatter(
            x=list(range(num_weeks+1, num_weeks+len(predicted_prices)+1)),
            y=predicted_prices,
            mode="lines", name="2025 Predicted",
            line=dict(color="red", width=3, dash="dash"),
            marker=dict(size=5, color="red"),
            showlegend=True
        ))

        # ✅ Highlight predicted extension
        fig.add_vrect(
            x0=num_weeks, x1=num_weeks + len(predicted_prices),
            fillcolor="lightgray", opacity=0.3,
            layer="below", line_width=0
        )

        # ✅ Bold labels, ticks, and title
        fig.update_layout(
            title=f"{selected_com} – Spot Price Trend (2020–2025)",
            xaxis_title="Weeks",
            yaxis_title="Price",
            xaxis_title_font=dict(size=14, color="black", family="Arial", weight="bold"),
            yaxis_title_font=dict(size=14, color="black", family="Arial", weight="bold"),
            xaxis=dict(tickfont=dict(size=12, color="black", family="Arial", weight="bold")),
            yaxis=dict(tickfont=dict(size=12, color="black", family="Arial", weight="bold")),
            title_font=dict(size=18, color="black", family="Arial", weight="bold"),
            template=custom_template,
            paper_bgcolor=PLOT_BGCOLOR,
            plot_bgcolor=PLOT_BGCOLOR,
            legend=dict(font=dict(size=8, color="black")),
            font=dict(color="black")
        )

        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("Not enough 2025 data for prediction.")



df_c = pd.read_excel(r"c:\Users\abdul\OneDrive\Desktop\price data.xlsx")
gdf = gpd.read_file(r"c:\Users\abdul\OneDrive\Desktop\Punjab_district_boundary.geojson")

# Clean Data
df_c = df_c.loc[:, ~df_c.columns.str.contains("^Unnamed")]
df_c.rename(columns={df_c.columns[0]: "District"}, inplace=True)
df_c["District"] = df_c["District"].str.lower().str.strip()

for col in df_c.columns[1:]:
    df_c[col] = pd.to_numeric(df_c[col], errors='coerce').fillna(np.nan)

gdf.rename(columns={col: "District" for col in gdf.columns if col.lower() in ["district", "district_name"]}, inplace=True)
gdf["District"] = gdf["District"].str.lower().str.strip()

# ✅ Use the same dropdown (selected_com) from above
if selected_com in df_c.columns:  # Only proceed if selected commodity exists in district data
    temp = df_c[["District", selected_com]].copy()
    temp.rename(columns={selected_com: "Price"}, inplace=True)
    temp["Price"] = pd.to_numeric(temp["Price"], errors='coerce')

    merged = gdf.merge(temp, on="District", how="left")
    merged["Price"] = merged["Price"].fillna(temp["Price"].mean())

    min_price = merged["Price"].min()
    max_price = merged["Price"].max()

    col1, col2 = st.columns([1.5, 1])

    with col1:
        fig_map = px.choropleth(
            merged,
            geojson=merged.__geo_interface__,
            locations="District",
            featureidkey="properties.District",
            color="Price",
            color_continuous_scale="armyrose",
            range_color=[min_price, max_price],
            hover_name="District",
            hover_data={"Price": ":.1f"},
            template=custom_template
        )

        merged["centroid"] = merged.geometry.centroid
        merged["lon"] = merged.centroid.x
        merged["lat"] = merged.centroid.y
        merged["label"] = merged["District"].str.title() + "<br>" + merged["Price"].round(1).astype(str)

        fig_map.add_trace(go.Scattergeo(
            lon=merged["lon"],
            lat=merged["lat"],
            text=merged["label"],
            mode="text",
            textfont=dict(color="black", size=10, family="Arial"),
            hoverinfo="skip"
        ))

        fig_map.update_geos(
            fitbounds="locations",
            visible=False,
            projection_type="mercator",
            center={"lat": 31.0, "lon": 73.0},
            projection_scale=1
        )

        fig_map.update_layout(
            title=f"🗺️ Average spot price(2025)_ {selected_com}",
            margin={"r":0,"t":40,"l":0,"b":0},
            height=800,
            coloraxis_showscale=False,
            paper_bgcolor=PLOT_BGCOLOR,
            plot_bgcolor=PLOT_BGCOLOR,
            title_font=dict(size=16, color="black"),
            font=dict(size=12, color="black")
        )
        st.plotly_chart(fig_map, use_container_width=True)

    with col2:
        # Sort by price in descending order for the bar chart
        bar_df = temp.sort_values("Price", ascending=False).copy()
        bar_df["Price"] = bar_df["Price"].fillna(temp["Price"].mean())
        bar_df["District"] = bar_df["District"].str.title()

        fig_bar = px.bar(
            bar_df,
            x="Price", 
            y="District",
            orientation="h",
            text=bar_df["Price"].round(1),
            color="Price", 
            color_continuous_scale="armyrose",
            range_color=[min_price, max_price],
            template=custom_template
        )

        fig_bar.update_traces(
            textposition="outside",
            textfont=dict(size=10, color="black")
        )
        fig_bar.update_layout(
            title=f"📊 Average spot price(2025) _ {selected_com}",
            yaxis=dict(
                categoryorder="total ascending",  # This ensures proper sorting by price
                tickfont=dict(size=10, color="black"),
                title_font=dict(size=12, color="black")
            ),
            xaxis=dict(
               
                title_font=dict(size=12, color="black"),
                tickfont=dict(color="black")
            ),
            margin={"r":0,"t":40,"l":0,"b":0},
            height=800,
            coloraxis_showscale=False,
            paper_bgcolor=PLOT_BGCOLOR, 
            plot_bgcolor=PLOT_BGCOLOR,
            title_font=dict(size=16, color="black"),
            font=dict(color="black"),
            uniformtext_minsize=8,
            uniformtext_mode='hide'
        )
        st.plotly_chart(fig_bar, use_container_width=True)
else:
    st.warning(f"No district data available for {selected_com}")