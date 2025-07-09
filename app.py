from flask import Flask, request, render_template, send_from_directory
import pandas as pd
import numpy as np
import pyreadr
import geopandas as gpd
from pyproj import CRS, Transformer
import plotly.express as px
import folium
from folium.plugins import MarkerCluster
import os
from datetime import datetime
from train import load_data, train_models

app = Flask(__name__)

# Define CRS
crs_bng = CRS("+proj=tmerc +lat_0=49 +lon_0=-2 +k=0.999601272 +x_0=400000 +y_0=-100000 +ellps=airy +units=km +no_defs")
crs_wgs84 = CRS("EPSG:4326")
transformer = Transformer.from_crs(crs_bng, crs_wgs84, always_xy=True)

# Global variables to store data and results
data = {"sitedata": None, "grid_data": None}
results = {"Random Forest": None, "Gaussian Process": None}
years_available = []

@app.route('/static/<path:filename>')
def serve_static(filename):
    return send_from_directory('static', filename)

@app.route("/", methods=["GET", "POST"])
def index():
    global data, results, years_available
    debug_message = "Waiting for file uploads and model execution..."

    if request.method == "POST":
        sitedata_file = request.files.get("sitedata_file")
        griddata_file = request.files.get("griddata_file")
        years = request.form.getlist("years")
        run_model = "run" in request.form

        if sitedata_file and griddata_file:
            try:
                upload_dir = os.path.join(app.root_path, "uploads")
                os.makedirs(upload_dir, exist_ok=True)
                sitedata_path = os.path.join(upload_dir, "London_NO2_Sitedata_FINAL.RData")
                griddata_path = os.path.join(upload_dir, "London_NO2_Griddata_FINAL.RData")
                sitedata_file.save(sitedata_path)
                griddata_file.save(griddata_path)

                sitedata, grid_data = load_data(sitedata_path, griddata_path)
                data = {"sitedata": sitedata, "grid_data": grid_data}
                years_available = sorted(sitedata["Year"].astype(float).unique())
                debug_message = f"Data loaded successfully. Years available: {', '.join(map(str, years_available))}"
            except Exception as e:
                debug_message = f"Error loading data: {str(e)}"
                return render_template("index.html", debug_message=debug_message, years_available=years_available)

        if run_model and data["sitedata"] is not None and years:
            try:
                debug_message += f"\nRunning Random Forest and Gaussian Process models for years: {', '.join(years)}"
                results = train_models(data["sitedata"], data["grid_data"])
                debug_message += "\nBoth models completed."
            except Exception as e:
                debug_message += f"\nError running models: {str(e)}"
                return render_template("index.html", debug_message=debug_message, years_available=years_available)

    # Generate plots
    plots = {"Random Forest": {"pred_plot": None, "uncertainty_plot": None},
             "Gaussian Process": {"pred_plot": None, "uncertainty_plot": None}}
    scatter_plot = None
    time_series_plot = None
    map_html = None
    error_message = ""

    for model_name in results:
        if results[model_name] is not None and results[model_name]["grid_data"] is not None and years:
            try:
                selected_years = [float(y) for y in years]
                filtered_data = results[model_name]["grid_data"][results[model_name]["grid_data"]["Year"].isin(selected_years)]
                debug_message += f"\nFiltered {model_name} data rows: {filtered_data.shape[0]}"
                filtered_data = filtered_data.dropna(subset=["NO2_pred", "NO2_sd"])
                filtered_data = filtered_data[~np.isinf(filtered_data["NO2_pred"]) & ~np.isinf(filtered_data["NO2_sd"])]
                if not filtered_data.empty:
                    plots[model_name]["pred_plot"] = px.density_heatmap(
                        filtered_data,
                        x="Easting",
                        y="Northing",
                        z="NO2_pred",
                        facet_col="Year",
                        facet_col_wrap=3,
                        color_continuous_scale="Viridis",
                        title=f"Predicted NO₂ Concentration ({model_name}, 2014–2019)",
                        labels={"NO2_pred": "NO₂ (µg/m³)", "Easting": "Easting", "Northing": "Northing"},
                        height=600
                    ).update_layout(xaxis=dict(scaleanchor="y", scaleratio=1)).to_json()

                    plots[model_name]["uncertainty_plot"] = px.density_heatmap(
                        filtered_data,
                        x="Easting",
                        y="Northing",
                        z="NO2_sd",
                        facet_col="Year",
                        facet_col_wrap=3,
                        color_continuous_scale="Magma",
                        title=f"Prediction Uncertainty ({model_name}, 2014–2019)",
                        labels={"NO2_sd": "Uncertainty (SD µg/m³)", "Easting": "Easting", "Northing": "Northing"},
                        height=600
                    ).update_layout(xaxis=dict(scaleanchor="y", scaleratio=1)).to_json()
                else:
                    error_message += f"No valid data for selected years {selected_years} in {model_name} heatmaps."
            except Exception as e:
                error_message += f"Error rendering {model_name} heatmaps: {str(e)}"

    if data["sitedata"] is not None:
        try:
            sitedata_clean = data["sitedata"].dropna(subset=["NO2", "NDVI"])
            sitedata_clean = sitedata_clean[~np.isinf(sitedata_clean["NO2"]) & ~np.isinf(sitedata_clean["NDVI"])]
            print(f"Shape of sitedata_clean: {sitedata_clean.shape}")  # Débogage
            print(f"Sample data: {sitedata_clean[['NO2', 'NDVI']].head().to_dict()}")  # Débogage
            if not sitedata_clean.empty:
                scatter_plot = px.scatter(
                    sitedata_clean,
                    x="NO2",
                    y="NDVI",
                    trendline="ols",
                    title="Scatterplot NO2 vs NDVI",
                    labels={"NO2": "NO₂ (µg/m³)", "NDVI": "NDVI"}
                ).to_json()
            else:
                error_message += "No valid data for scatterplot after cleaning."
        except Exception as e:
            error_message += f"Error rendering scatterplot: {str(e)}"

        try:
            time_series = data["sitedata"].groupby("Year")["NO2"].mean().reset_index()
            time_series = time_series.dropna()
            time_series = time_series[~np.isinf(time_series["NO2"])]
            if not time_series.empty:
                time_series_plot = px.line(
                    time_series,
                    x="Year",
                    y="NO2",
                    title="Evolution of Average NO2 per Year",
                    labels={"NO2": "NO₂ (µg/m³)", "Year": "Year"}
                ).to_json()
            else:
                error_message += "No valid data for time series plot."
        except Exception as e:
            error_message += f"Error rendering time series: {str(e)}"

        try:
            sites = data["sitedata"][["siteid", "Easting", "Northing"]].drop_duplicates()
            sites_gdf = gpd.GeoDataFrame(sites, geometry=gpd.points_from_xy(sites["Easting"], sites["Northing"]), crs=crs_bng)
            sites_gdf = sites_gdf.to_crs(crs_wgs84)
            m = folium.Map(location=[51.5074, -0.1278], zoom_start=10)
            marker_cluster = MarkerCluster().add_to(m)
            for idx, row in sites_gdf.iterrows():
                folium.Marker([row.geometry.y, row.geometry.x], popup=row["siteid"]).add_to(marker_cluster)
            map_html = m._repr_html_()
        except Exception as e:
            error_message += f"Error rendering map: {str(e)}"

    if error_message:
        debug_message += f"\nErrors: {error_message}"

    return render_template("index.html", debug_message=debug_message, years_available=years_available,
                           results=results, plots=plots, scatter_plot=scatter_plot,
                           time_series_plot=time_series_plot, map_html=map_html)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)