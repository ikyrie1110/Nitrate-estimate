# 🔬 Data-Driven Pollution Estimation and Model Interpretability Platform

This project consists of two complementary modules:

- **`variable_selection_and_heatmap/`**: For feature selection and correlation analysis
- **`data_processing_and_plotting/`**: For model training, SHAP analysis, and spatial visualization

---

## 📁 Project Structure

```
├── variable_selection_and_heatmap/         # Feature selection and correlation analysis
│   ├── data_preprocessing.py              # Data loading and preprocessing
│   ├── feature_selection.py               # Feature importance analysis using multiple models
│   ├── heatmap.py                         # Correlation heatmap plotting
│   ├── model_training.py                  # Training models and evaluating performance
│   └── main.py                            # Main entry for the full feature analysis workflow

├── data_processing_and_plotting/          # Model training, SHAP analysis, and visualization
│   ├── data_preprocessing.py              # Data cleaning
│   ├── model_training.py                  # Training individual models (RF, XGB, CatBoost, GBDT)
│   ├── stacking.py                        # Stacked feature generation and ElasticNet meta-model
│   ├── shap_analysis.py                   # SHAP analysis for base and stacked models
│   ├── plotting.py                        # GeoTIFF rendering and boundary plotting
│   └── main.py                            # Entry point for training, SHAP, and map generation

└── requirements.txt                       # Python dependencies
```

---

## 🧠 Module Overview

| Module Path | Description |
|-------------|-------------|
| `variable_selection_and_heatmap/` | Feature importance assessment, heatmap plotting, and model-based ranking |
| `data_processing_and_plotting/`   | Pollution regression estimation, SHAP explainability, and mapping outputs |

---

## 🚀 Recommended Usage

```bash
# Step 1: Feature selection and correlation visualization
cd variable_selection_and_heatmap
python main.py

# Step 2: Model training, SHAP analysis, and map visualization
cd data_processing_and_plotting
python main.py
```

---

## 📦 Installation

Install the required dependencies:

```bash
pip install -r requirements.txt
```

Key dependencies:
- `pandas`, `numpy`, `scikit-learn`, `matplotlib`, `shap`
- `xgboost`, `catboost`, `rasterio`, `basemap`, `shapely`

---

## 📊 Results and Output

- The best trained model will be saved as a `.joblib` file.
- SHAP analysis results will be displayed showing the feature importance.
- The stacked ensemble model's performance (e.g., RMSE, R²) will be output to the console.
- Maps and other visualizations will be saved as image files.

---

## 📂 Data Availability

The data required for this project (such as `XXX.csv` and other inputs) will be provided upon request.  
Please contact the project maintainers to access the datasets.

---

## 🤝 Contributions

Feel free to raise any issues or suggest improvements!  
If you have any questions, don’t hesitate to open an issue or submit a pull request.
