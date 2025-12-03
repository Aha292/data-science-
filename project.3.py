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
BG_COLOR = "#f0f2f6"
PLOT_BGCOLOR = "#FFFFFF"
FONT_COLOR = "#000000"

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

# Custom template
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
            "title_standoff": 15
        },
        "yaxis": {
            "gridcolor": "#e6e6e6",
            "linecolor": "#e6e6e6",
            "zerolinecolor": "#e6e6e6",
            "title_font": {"size": 14, "color": "black"},
            "tickfont": {"color": "black"},
            "title_standoff": 15
        }
    }
}

# -----------------------
# Case 1: Potato
# -----------------------
if selected_com == "Potato":
    df_2025_actual = df[df["spot_price_2025"] > 0].copy().reset_index(drop=True)

    seasonal_prices = []
    for year in ["2020", "2021", "2022", "2023", "2024"]:
        prices = df[f"spot_price_{year}"].values[:52]
        for week, price in enumerate(prices, 1):
            seasonal_prices.append([week, price])
    season_df = pd.DataFrame(seasonal_prices, columns=["Week", "Price"])
    avg_season = season_df.groupby("Week")["Price"].mean()

    all_data = []
    for year in ["2020", "2021", "2022", "2023", "2024"]:
        prices = df[f"spot_price_{year}"].values[:52]
        for week, price in enumerate(prices, 1):
            all_data.append([week, price, year, "Actual"])

    # Actual 2025
    actual_2025 = df["spot_price_2025"][df["spot_price_2025"] > 0].values
    for week, price in enumerate(actual_2025, 1):
        all_data.append([week, price, "2025", "Actual"])

    # Predicted 2025 (week 30 onward)
    for week in range(30, 53):
        predicted_price = avg_season[week]
        all_data.append([week, predicted_price, "2025", "Predicted"])

    plot_data = pd.DataFrame(all_data, columns=["Week", "Price", "Year", "Type"])

    # Base years
    fig_full = px.line(
        plot_data[(plot_data["Type"] == "Actual") & (plot_data["Year"] != "2025")],
        x="Week", y="Price", color="Year",
        title="Trend of spot price – Potato (2020–2025)",
        template=custom_template,
        labels={"Week": "Weeks", "Price": "Price"},
        color_discrete_sequence=px.colors.qualitative.Bold
    )

    # --- Actual 2025 solid ---
    actual_25 = plot_data[(plot_data["Year"] == "2025") & (plot_data["Type"] == "Actual")]
    fig_full.add_trace(go.Scatter(
        x=actual_25["Week"],
        y=actual_25["Price"],
        mode="lines",
        name="2025 Actual",
        line=dict(color="black", width=3),
        showlegend=True
    ))

    # --- Predicted 2025 dashed (connected) ---
    pred_25 = plot_data[(plot_data["Year"] == "2025") & (plot_data["Type"] == "Predicted")]
    if not actual_25.empty and not pred_25.empty:
        fig_full.add_trace(go.Scatter(
            x=[actual_25["Week"].iloc[-1]] + pred_25["Week"].tolist(),
            y=[actual_25["Price"].iloc[-1]] + pred_25["Price"].tolist(),
            mode="lines",
            name="2025 Prediction",
            line=dict(color="black", width=3, dash="dash"),
            showlegend=True
        ))

    # Highlight prediction area
    fig_full.add_vrect(
        x0=30, x1=52,
        fillcolor="lightgray", opacity=0.3,
        layer="below", line_width=0
    )

    # Layout
    fig_full.update_layout(
        xaxis=dict(
            title="Weeks", range=[1, 52],
            tickmode="linear", dtick=1,
            title_font=dict(size=14, color="black", family="Arial",),
            tickfont=dict(size=12, color="black", family="Arial",)
        ),
        yaxis=dict(
            title="Price in Rupees",
            title_font=dict(size=14, color="black", family="Arial",),
            tickfont=dict(size=12, color="black", family="Arial",)
        ),
        title_font=dict(size=18, color="black", family="Arial",),
        paper_bgcolor="white",
        plot_bgcolor="white",
        legend=dict(
            orientation="h",
            yanchor="bottom", y=-0.25,
            xanchor="center", x=0.5,
            bgcolor="rgba(255,255,255,0.8)",
            bordercolor="black", borderwidth=1,
            font=dict(size=11, color="black", family="Arial Bold")
        )
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
        # Train linear regression
        x_train = np.arange(num_weeks).reshape(-1, 1)
        y_train = df_2025["spot_price_2025"]
        model = LinearRegression().fit(x_train, y_train)

        # Predict trend
        x_pred = np.arange(num_weeks, num_weeks + 20).reshape(-1, 1)
        trend_pred = model.predict(x_pred)

        # Seasonal averages
        seasonal_prices = []
        for yr in ["2020","2021","2022","2023","2024"]:
            if f"spot_price_{yr}" in cdf:
                prices = cdf[f"spot_price_{yr}"].values[:52]
                if len(prices) < 52:
                    prices = np.append(prices, [np.nan]*(52-len(prices)))
                seasonal_prices.extend(prices)
        seasonal_matrix = np.array(seasonal_prices).reshape(5, 52)
        avg_season = np.nanmean(seasonal_matrix, axis=0)

        # Blend trend + seasonal prediction
        predicted_prices = []
        for i in range(20):
            wk = num_weeks + i
            season_wk = wk % 52
            combined = (trend_pred[i] + avg_season[season_wk]) / 2
            predicted_prices.append(round(combined, 2))

        # --- Build Figure ---
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

        # --- Actual 2025 solid ---
        fig.add_trace(go.Scatter(
            x=list(range(1, num_weeks+1)),
            y=df_2025["spot_price_2025"],
            mode="lines",
            name="2025 Actual",
            line=dict(color="black", width=3),
            showlegend=True
        ))

        # --- Predicted 2025 dashed (connected) ---
        fig.add_trace(go.Scatter(
            x=[num_weeks] + list(range(num_weeks+1, num_weeks+len(predicted_prices)+1)),
            y=[df_2025["spot_price_2025"].iloc[-1]] + predicted_prices,
            mode="lines",
            name="2025 Prediction",
            line=dict(color="black", width=3, dash="dash"),
            showlegend=True
        ))

        # Highlight prediction area
        fig.add_vrect(
            x0=num_weeks, x1=num_weeks + len(predicted_prices),
            fillcolor="lightgray", opacity=0.3,
            layer="below", line_width=0
        )

        # Layout
        fig.update_layout(
            title=f"{selected_com} – Spot Price Trend (2020–2025)",
            xaxis=dict(
                title="Weeks", range=[1, 52],
                tickmode="linear", dtick=1,
                title_font=dict(size=14, color="black", family="Arial"),
                tickfont=dict(size=12, color="black", family="Arial")
            ),
            yaxis=dict(
                title="Price in Rupees",
                title_font=dict(size=14, color="black", family="Arial"),
                tickfont=dict(size=12, color="black", family="Arial")
            ),
            title_font=dict(size=18, color="black", family="Arial"),
            template=custom_template,
            paper_bgcolor=PLOT_BGCOLOR,
            plot_bgcolor=PLOT_BGCOLOR,
            legend=dict(
                orientation="h",
                yanchor="bottom", y=-0.25,
                xanchor="center", x=0.5,
                font=dict(size=14, color="black", family="Arial Bold")
            )
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("Not enough 2025 data for prediction.")

# -----------------------
# District-wise map + bar
# -----------------------
df_c = pd.read_excel(r"c:\Users\abdul\OneDrive\Desktop\price data.xlsx")
gdf = gpd.read_file(r"c:\Users\abdul\OneDrive\Desktop\Punjab_district_boundary.geojson")

df_c = df_c.loc[:, ~df_c.columns.str.contains("^Unnamed")]
df_c.rename(columns={df_c.columns[0]: "District"}, inplace=True)
df_c["District"] = df_c["District"].str.lower().str.strip()

for col in df_c.columns[1:]:
    df_c[col] = pd.to_numeric(df_c[col], errors='coerce').fillna(np.nan)

gdf.rename(columns={col: "District" for col in gdf.columns if col.lower() in ["district", "district_name"]}, inplace=True)
gdf["District"] = gdf["District"].str.lower().str.strip()

if selected_com in df_c.columns:
    temp = df_c[["District", selected_com]].copy()
    temp.rename(columns={selected_com: "Price"}, inplace=True)
    temp["Price"] = pd.to_numeric(temp["Price"], errors='coerce')

    merged = gdf.merge(temp, on="District", how="left")
    merged["Price"] = merged["Price"].fillna(temp["Price"].mean())

    min_price = merged["Price"].min()
    max_price = merged["Price"].max()

    # Create shared colorscale
    shared_colorscale = "armyrose"
    
    # Create 5 bins for the legend
    bins = np.linspace(min_price, max_price, 6)  # 5 intervals
    color_scale_vals = px.colors.get_colorscale(shared_colorscale)
    colors = px.colors.sample_colorscale(color_scale_vals, [i/5 for i in range(5)])
    
    # Create the shared legend
    legend_fig = go.Figure()
    
    for i in range(5):
        legend_fig.add_trace(go.Scatter(
            x=[None], y=[None],
            mode="markers",
            marker=dict(size=15, color=colors[i], symbol="circle"),
            name=f"{bins[i]:.1f} - {bins[i+1]:.1f}",
            hoverinfo="none"
        ))
    
    legend_fig.update_layout(
        title="Price Scale",
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="center",
            x=0.5,
            font=dict(size=10, color="black"),
            itemsizing="constant"
        ),
        margin=dict(l=0, r=0, t=30, b=0),
        height=50,
        paper_bgcolor=PLOT_BGCOLOR,
        plot_bgcolor=PLOT_BGCOLOR
    )
    
    # Display the legend above the charts
    st.plotly_chart(legend_fig, use_container_width=True)

    # ---------------- Map and Bar Charts ----------------
    col1, col2 = st.columns([1.5, 1])
    
    with col1:
        fig_map = px.choropleth(
            merged,
            geojson=merged.__geo_interface__,
            locations="District",
            featureidkey="properties.District",
            color="Price",
            color_continuous_scale=shared_colorscale,
            range_color=[min_price, max_price],
            hover_name="District",
            hover_data={"Price": ":.1f"},
            template=custom_template
        )
        fig_map.update_layout(coloraxis_showscale=False)

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
            height=600,
            paper_bgcolor=PLOT_BGCOLOR,
            plot_bgcolor=PLOT_BGCOLOR,
            title_font=dict(size=16, color="black"),
            font=dict(size=12, color="black")
        )
        st.plotly_chart(fig_map, use_container_width=True)

    with col2:
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
            color_continuous_scale=shared_colorscale,
            range_color=[min_price, max_price],
            template=custom_template
        )
        fig_bar.update_layout(coloraxis_showscale=False)

        fig_bar.update_traces(
            textposition="outside",
            textfont=dict(size=10, color="black")
        )
        fig_bar.update_layout(
            title=f"📊 Average spot price(2025) _ {selected_com}",
            yaxis=dict(
                categoryorder="total ascending",
                tickfont=dict(size=10, color="black"),
                title_font=dict(size=12, color="black")
            ),
            xaxis=dict(
                title_font=dict(size=12, color="black"),
                tickfont=dict(color="black")
            ),
            margin={"r":10,"t":40,"l":0,"b":0},
            height=600,
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
