import streamlit as st
import pandas as pd
import numpy as np
import geopandas as gpd
import plotly.express as px
import plotly.graph_objects as go
import json
from sklearn.linear_model import LinearRegression

st.set_page_config(layout="wide", page_title="Punjab Potato Price Dashboard")

st.title("Punjab Potato Spot Price Dashboard")

# -----------------------
# Part 1: Time-Series Data
# -----------------------

df = pd.read_csv(r"c:\Users\abdul\Downloads\amCharts (2).csv")
df = df.drop(['spot_price_2026', 'spot_price_2027'], axis=1)
df["Date"] = pd.to_datetime(df["Date"])

df_2025_actual = df[df["spot_price_2025"] > 0].copy()
x_train = df_2025_actual.index.values.reshape(-1, 1)
y_train = df_2025_actual["spot_price_2025"]
model = LinearRegression().fit(x_train, y_train)

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

actual_2025 = df["spot_price_2025"][df["spot_price_2025"] > 0].values
for week, price in enumerate(actual_2025, 1):
    all_data.append([week, price, "2025", "Actual"])

for week in range(30, 53):
    predicted_price = avg_season.get(week, np.nan)
    all_data.append([week, predicted_price, "2025", "Predicted"])

plot_data = pd.DataFrame(all_data, columns=["Week", "Price", "Year", "Type"])

# Full multi-year chart
fig_full = px.line(
    plot_data[(plot_data["Type"] == "Actual") & (plot_data["Year"] != "2025")],
    x="Week", y="Price", color="Year",
    title="Full Potato Spot Prices (2020–2025)",
    template="plotly_white",
    labels={"Week": "Week of Year", "Price": "Spot Price"}
)
combined_2025 = plot_data[plot_data["Year"] == "2025"]
fig_full.add_scatter(
    x=combined_2025["Week"],
    y=combined_2025["Price"],
    mode="lines+markers",
    name="2025 (Actual + Predicted)",
    line=dict(color="red", width=3),
    marker=dict(size=5)
)
fig_full.add_vrect(
    x0=29.5, x1=52,
    fillcolor="lightgray", opacity=0.25,
    annotation_text="Predicted Region (2025)",
    annotation_position="top left",
    line_width=0
)
fig_full.update_layout(
    hovermode="x unified",
    legend_title_text="Year",
    xaxis=dict(tickmode='linear', dtick=4, range=[1, 52]),
    yaxis=dict(title="Price", range=[0, plot_data["Price"].max() * 1.1]),
    height=600
)

# 2025 only chart
combined_2025_data = plot_data[plot_data["Year"] == "2025"].sort_values("Week")
fig_2025 = go.Figure()
fig_2025.add_trace(go.Scatter(
    x=combined_2025_data["Week"],
    y=combined_2025_data["Price"],
    mode="lines+markers",
    name="2025 (Actual + Predicted)",
    line=dict(color="blue", width=3),
    marker=dict(size=6)
))
fig_2025.add_vrect(
    x0=29.5, x1=52,
    fillcolor="lightgray", opacity=0.25,
    annotation_text="Predicted Region",
    annotation_position="top left",
    line_width=0
)
fig_2025.update_layout(
    title="2025 Only: Potato Spot Prices (Actual + Predicted)",
    xaxis=dict(title="Week of Year", tickmode='linear', dtick=2),
    yaxis=dict(title="Spot Price", range=[0, plot_data["Price"].max() * 1.1]),
    hovermode="x unified",
    legend=dict(title=""),
    template="plotly_white",
    height=600
)

# Prediction table
week_labels = [f"2025-{wk}" for wk in range(30, 53)]
future_dates = pd.date_range("2025-07-21", periods=23, freq="W-MON")
predicted_prices = [round(avg_season.get(wk, np.nan), 1) for wk in range(30, 53)]
future_df = pd.DataFrame({
    "Week": week_labels,
    "Date": future_dates.strftime("%Y-%m-%d"),
    "Predicted Price (2025)": predicted_prices
})

# -----------------------
# Part 2: District-Level Commodities
# -----------------------

df_dist = pd.read_excel(r"c:\Users\abdul\OneDrive\Desktop\price data.xlsx")
gdf = gpd.read_file(r"c:\Users\abdul\OneDrive\Desktop\Punjab_district_boundary.geojson")

df_dist = df_dist.loc[:, ~df_dist.columns.str.contains("^Unnamed")]
df_dist.rename(columns={df_dist.columns[0]: "District"}, inplace=True)
df_dist["District"] = df_dist["District"].str.strip().str.lower()

# Rename district column in gdf to "District" lowercase for merge
gdf_cols_lower = [col.lower() for col in gdf.columns]
for col in gdf.columns:
    if col.lower() in ["district", "district_name"]:
        gdf.rename(columns={col: "District"}, inplace=True)
        break
gdf["District"] = gdf["District"].str.strip().str.lower()

commodities = df_dist.columns.drop("District").tolist()

color_scale = [
    [0.0, "#d4e6f1"],
    [0.25, "#a9cce3"],
    [0.5, "#5dade2"],
    [0.75, "#2e86c1"],
    [1.0, "Grey"]
]

plots = {}
for commodity in commodities:
    temp = df_dist[["District", commodity]].copy()
    temp.rename(columns={commodity: "Price"}, inplace=True)
    temp["Price"] = pd.to_numeric(temp["Price"], errors='coerce').fillna(temp["Price"].mean())
    temp["hover_text"] = temp.apply(
        lambda row: f"{row['District'].title()}<br>{commodity}: {row['Price']:.1f}", axis=1
    )

    merged = gdf.merge(temp, on="District", how="left")
    merged["Price"] = merged["Price"].fillna(temp["Price"].mean())
    merged["hover_text"] = merged["hover_text"].fillna("No Data")

    geojson_data = json.loads(merged.to_json())
    merged["centroid"] = merged.geometry.centroid
    merged["lon"] = merged["centroid"].x
    merged["lat"] = merged["centroid"].y

    # Price labels as string rounded 1 decimal
    price_labels = merged["Price"].round(1).astype(str).fillna("NA")

    # Create combined labels: "District\nPrice"
    combined_labels = merged.apply(lambda r: f"{r['District'].title()}\n{r['Price']:.1f}", axis=1)

    min_val, max_val = temp["Price"].min(), temp["Price"].max()
    bins = np.linspace(min_val, max_val, 5)
    bin_labels = [f"{bins[i]:.1f} – {bins[i+1]:.1f}" for i in range(len(bins)-1)]
    legend_colors = ["#d4e6f1", "#a9cce3", "#5dade2", "#2e86c1"]

    legend_traces = [
        go.Scattergeo(lon=[None], lat=[None], mode='markers',
                      marker=dict(size=14, color=color, symbol='square'),
                      name=label, showlegend=True)
        for color, label in zip(legend_colors, bin_labels)
    ]

    map_trace = go.Choropleth(
        geojson=geojson_data,
        locations=merged["District"],
        z=merged["Price"],
        featureidkey="properties.District",
        colorscale=color_scale,
        marker_line_width=0.5,
        hovertext=merged["hover_text"],
        hoverinfo="text",
        showscale=False,
        zmin=min_val,
        zmax=max_val
    )

    label_bg_trace = go.Scattergeo(
        lon=merged["lon"],
        lat=merged["lat"],
        mode='markers',
        marker=dict(size=16, color="white", opacity=0.9, line=dict(color='black', width=0.5)),
        hoverinfo='none',
        showlegend=False
    )

    # Add combined label trace with district + price, normal font (not bold)
    label_text_trace = go.Scattergeo(
        lon=merged["lon"],
        lat=merged["lat"],
        mode='text',
        text=combined_labels,
        textfont=dict(size=10, color="black", family="Arial"),
        textposition='middle center',
        hoverinfo='none',
        showlegend=False
    )

    map_layout = dict(
        title=f"{commodity.title()} Price by District",
        geo=dict(fitbounds="locations", visible=False),
        margin=dict(r=20, t=120, l=20, b=20),
        height=900,
        legend=dict(
            title="Range",
            orientation="h",
            yanchor="bottom",
            y=1.12,
            xanchor="center",
            x=0.5,
            bgcolor="rgba(255,255,255,0.95)",
            bordercolor="black",
            borderwidth=1
        )
    )

    bar_df = temp.sort_values("Price", ascending=False)
    bar_trace = go.Bar(
        x=bar_df["Price"],
        y=bar_df["District"].str.title(),
        orientation='h',
        marker=dict(
            color=bar_df["Price"],
            colorscale=color_scale,
            line=dict(color='black', width=1.5)
        ),
        hovertext=bar_df["hover_text"],
        hoverinfo="text",
        text=[f"{p:.1f}" for p in bar_df["Price"]],
        textposition="auto",
        showlegend=False
    )

    bar_layout = dict(
        title=f"{commodity.title()} Price Across All Districts",
        xaxis_title="Price",
        yaxis_title="District",
        height=900,
        margin=dict(l=220, r=20, t=50, b=50),
        legend=dict(
            title="Range",
            orientation="h",
            yanchor="bottom",
            y=1.12,
            xanchor="center",
            x=0.5,
            bgcolor="rgba(255,255,255,0.95)",
            bordercolor="black",
            borderwidth=1
        )
    )

    plots[commodity] = {
        "map": go.Figure(data=[map_trace, label_bg_trace, label_text_trace] + legend_traces, layout=map_layout),
        "bar": go.Figure(data=[bar_trace], layout=bar_layout)
    }

# -----------------------
# Streamlit UI
# -----------------------

# Show time-series charts
st.plotly_chart(fig_full, use_container_width=True)
st.plotly_chart(fig_2025, use_container_width=True)

# Show prediction table
