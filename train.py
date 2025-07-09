import pandas as pd
import numpy as np
import pyreadr
from sklearn.ensemble import RandomForestRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import pickle
import os

def load_data(sitedata_path="uploads/London_NO2_Sitedata_FINAL.RData", griddata_path="uploads/London_NO2_Griddata_FINAL.RData"):
    """Load and preprocess site and grid data."""
    try:
        print(f"Checking file: {os.path.abspath(sitedata_path)}")
        if not os.path.exists(sitedata_path):
            raise FileNotFoundError(f"File {sitedata_path} does not exist!")
        print(f"Checking file: {os.path.abspath(griddata_path)}")
        if not os.path.exists(griddata_path):
            raise FileNotFoundError(f"File {griddata_path} does not exist!")
        
        sitedata = pyreadr.read_r(sitedata_path)["sitedata"]
        grid_data = pyreadr.read_r(griddata_path)["all"]

        if "Year" not in sitedata.columns:
            raise ValueError("Error: 'Year' column not found in sitedata.")
        if not pd.to_numeric(sitedata["Year"], errors="coerce").notnull().all():
            raise ValueError("Error: 'Year' column contains non-numeric values.")
        
        print(f"Before dropna: {sitedata.shape}")
        sitedata = sitedata.dropna(subset=["NO2", "NDVI", "Easting", "Northing", "Year"])
        print(f"After dropna: {sitedata.shape}")
        sitedata["date"] = pd.to_datetime(sitedata["Year"].astype(float).astype(int).astype(str))
        sitedata["time"] = pd.factorize(sitedata["Year"])[0] + 1
        sitedata["log_NO2"] = np.log(sitedata["NO2"] + 1)

        grid_data = grid_data[["Easting", "Northing", "Time", "NDVI", "gridid", "Year"]]
        grid_data = grid_data.rename(columns={"Time": "time"})
        grid_data["Year"] = grid_data["Year"].astype(float)

        return sitedata, grid_data
    except Exception as e:
        raise Exception(f"Error loading data: {str(e)}")

def train_models(sitedata, grid_data, output_dir="models"):
    """Train both Random Forest and Gaussian Process models."""
    try:
        os.makedirs(output_dir, exist_ok=True)
        results = {}

        # Random Forest
        train_data = sitedata[["log_NO2", "Easting", "Northing", "time", "NDVI"]]
        rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
        rf_model.fit(train_data[["Easting", "Northing", "time", "NDVI"]], train_data["log_NO2"])

        with open(os.path.join(output_dir, "rf_model.pkl"), "wb") as f:
            pickle.dump(rf_model, f)

        rf_grid_data = grid_data.copy()
        pred_data = rf_grid_data[["Easting", "Northing", "time", "NDVI"]]
        pred_all = rf_model.predict(pred_data)
        rf_grid_data["log_NO2_pred"] = pred_all
        rf_grid_data["log_NO2_sd"] = np.std([tree.predict(pred_data) for tree in rf_model.estimators_], axis=0)
        rf_grid_data["NO2_pred"] = np.exp(rf_grid_data["log_NO2_pred"])
        rf_grid_data["NO2_sd"] = rf_grid_data["NO2_pred"] * rf_grid_data["log_NO2_sd"]

        rf_cv_results = []
        kf = KFold(n_splits=5, shuffle=True, random_state=42)
        for fold, (train_idx, test_idx) in enumerate(kf.split(sitedata)):
            train_fold = sitedata.iloc[train_idx]
            test_fold = sitedata.iloc[test_idx]
            rf_fold = RandomForestRegressor(n_estimators=500, random_state=42)
            rf_fold.fit(train_fold[["Easting", "Northing", "time", "NDVI"]], train_fold["log_NO2"])
            pred_log = rf_fold.predict(test_fold[["Easting", "Northing", "time", "NDVI"]])
            pred = np.exp(pred_log)
            obs = test_fold["NO2"]
            rmse = np.sqrt(mean_squared_error(obs, pred))
            mae = mean_absolute_error(obs, pred)
            bias = np.mean(pred - obs)
            r2 = r2_score(obs, pred)
            rf_cv_results.append({"Fold": fold + 1, "RMSE": rmse, "MAE": mae, "Bias": bias, "R2": r2})
        rf_cv_summary = pd.DataFrame(rf_cv_results).mean().to_dict()

        results["Random Forest"] = {"grid_data": rf_grid_data, "cv_summary": rf_cv_summary}

        # Gaussian Process (INLA-SPDE Proxy)
        kernel = RBF(length_scale=10) + WhiteKernel(noise_level=1)
        gp_model = GaussianProcessRegressor(kernel=kernel, random_state=42)
        gp_model.fit(train_data[["Easting", "Northing", "time", "NDVI"]], train_data["log_NO2"])

        with open(os.path.join(output_dir, "gp_model.pkl"), "wb") as f:
            pickle.dump(gp_model, f)

        gp_grid_data = grid_data.copy()
        pred_data = gp_grid_data[["Easting", "Northing", "time", "NDVI"]]
        pred_all, pred_std = gp_model.predict(pred_data, return_std=True)
        gp_grid_data["log_NO2_pred"] = pred_all
        gp_grid_data["log_NO2_sd"] = pred_std
        gp_grid_data["NO2_pred"] = np.exp(gp_grid_data["log_NO2_pred"])
        gp_grid_data["NO2_sd"] = gp_grid_data["NO2_pred"] * gp_grid_data["log_NO2_sd"]

        gp_cv_results = []
        kf = KFold(n_splits=5, shuffle=True, random_state=23)
        for fold, (train_idx, test_idx) in enumerate(kf.split(sitedata)):
            train_fold = sitedata.iloc[train_idx]
            test_fold = sitedata.iloc[test_idx]
            gp_fold = GaussianProcessRegressor(kernel=kernel, random_state=42)
            gp_fold.fit(train_fold[["Easting", "Northing", "time", "NDVI"]], train_fold["log_NO2"])
            pred_log, _ = gp_fold.predict(test_fold[["Easting", "Northing", "time", "NDVI"]], return_std=True)
            pred = np.exp(pred_log)
            obs = test_fold["NO2"]
            rmse = np.sqrt(mean_squared_error(obs, pred))
            mae = mean_absolute_error(obs, pred)
            bias = np.mean(pred - obs)
            r2 = r2_score(obs, pred)
            gp_cv_results.append({"Fold": fold + 1, "RMSE": rmse, "MAE": mae, "Bias": bias, "R2": r2})
        gp_cv_summary = pd.DataFrame(gp_cv_results).mean().to_dict()

        results["Gaussian Process"] = {"grid_data": gp_grid_data, "cv_summary": gp_cv_summary}

        # Save results
        for model_name, model_results in results.items():
            model_results["grid_data"].to_csv(os.path.join(output_dir, f"{model_name.replace(' ', '_')}_predictions.csv"), index=False)
            pd.DataFrame([model_results["cv_summary"]]).to_csv(os.path.join(output_dir, f"{model_name.replace(' ', '_')}_cv_summary.csv"), index=False)

        return results
    except Exception as e:
        raise Exception(f"Error training models: {str(e)}")

if __name__ == "__main__":
    sitedata_path = "uploads/London_NO2_Sitedata_FINAL.RData"
    griddata_path = "uploads/London_NO2_Griddata_FINAL.RData"
    sitedata, grid_data = load_data(sitedata_path, griddata_path)
    results = train_models(sitedata, grid_data)
    print("Model training completed. Results saved in 'models' directory.")
