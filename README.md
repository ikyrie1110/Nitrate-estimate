
# Project Overview

This project consists of multiple modules that are primarily used for data loading, preprocessing, feature selection, model training, stacking ensemble, SHAP analysis, and correlation heatmap plotting. Below is a brief description of each module and instructions on how to run the project.

## Project Structure

```
project/
│
├── variable_selection_and_heatmap/         # Variable selection and heatmap plotting
│   ├── data_preprocessing.py              # Data loading and preprocessing
│   ├── feature_selection.py               # Feature selection and importance calculation
│   ├── heatmap.py                         # Correlation heatmap plotting
│   ├── model_training.py                  # Model training and evaluation
│   ├── main.py                            # Main program entry
│
└── data_processing_and_plotting/          # Data processing and plotting
    ├── data_preprocessing.py              # Data loading and preprocessing
    ├── model_training.py                  # Model training
    ├── stacking.py                        # Stacking ensemble
    ├── shap_analysis.py                   # SHAP analysis
    ├── main.py                            # Main program entry
    └── plotting.py                        # Plotting module
```

## Module Descriptions

### `variable_selection_and_heatmap` Folder

#### `data_preprocessing.py`
- **Function**: Loads the raw data, performs missing value handling, data cleaning, and preprocessing.
- **Key Functions**:
  - `load_data(data_path, importance_path)`: Loads the data and feature importance file.
  - `preprocess_data(data)`: Cleans the data by handling invalid values (e.g., `NaN`, `inf`), and performs data preprocessing.

#### `feature_selection.py`
- **Function**: Computes feature importance and filters important features based on a threshold.
- **Key Functions**:
  - `calculate_feature_importance(feature_importance_df, threshold=0.5)`: Computes and selects important features.

#### `heatmap.py`
- **Function**: Plots a correlation heatmap and shows significance annotations.
- **Key Functions**:
  - `plot_correlation_heatmap(data)`: Computes the correlation matrix and plots the heatmap with significance annotations.

#### `model_training.py`
- **Function**: Trains machine learning models and evaluates their performance.
- **Key Functions**:
  - `prepare_features_and_target(data)`: Prepares the features and target variables.
  - `split_data(X, y)`: Splits the data into training and test sets.
  - `train_models(X_train, y_train)`: Trains models and performs hyperparameter tuning using grid search.

#### `main.py`
- **Function**: Main program entry that calls all the modules and executes the full pipeline.
- **Key Functionality**:
  - Calls the functions from the above modules to perform data loading, preprocessing, feature selection, model training, evaluation, and heatmap plotting.

### `data_processing_and_plotting` Folder

#### `data_preprocessing.py`
- **Function**: Loads and preprocesses the data, cleans and transforms it into a suitable format for modeling.
- **Key Functions**:
  - `load_data(data_path)`: Loads the data file.
  - `preprocess_data(data)`: Cleans the data by handling invalid values, missing values, etc.

#### `model_training.py`
- **Function**: Trains different regression models such as Random Forest, XGBoost, CatBoost, etc.
- **Key Functions**:
  - `prepare_features_and_target(data)`: Extracts features and target variables from the data.
  - `train_models(X_train, y_train)`: Trains and evaluates different regression models.

#### `stacking.py`
- **Function**: Stacking ensemble method used to combine predictions from multiple base models.
- **Key Functions**:
  - `generate_stacking_features(X, y, models)`: Generates features for stacking models.
  - `stack_and_evaluate(X_train_stacking, y_train, X_test_stacking, y_test, meta_model)`: Trains the stacked model and evaluates its performance.

#### `shap_analysis.py`
- **Function**: Performs SHAP (Shapley Additive Explanations) analysis for model interpretability, showing feature importance.
- **Key Functions**:
  - `shap_analysis(best_model, X_train, X_test)`: Performs SHAP analysis on the best model and outputs feature importance.

#### `plotting.py`
- **Function**: Plots visualizations related to model training, such as maps, prediction results, etc.
- **Key Functions**:
  - `plot_geotiff_with_shapes(folder_path, output_folder)`: Plots GeoTIFF files with overlaid shape files.

#### `main.py`
- **Function**: Main program entry that integrates and runs data loading, feature selection, model training, stacking ensemble, SHAP analysis, and plotting.
- **Key Functionality**:
  - Calls the functions from the above modules to execute the full data processing and model evaluation pipeline.

## How to Run

1. **Install Dependencies**
   - This project uses the following major libraries:
     - `numpy`
     - `pandas`
     - `scikit-learn`
     - `xgboost`
     - `catboost`
     - `shap`
     - `matplotlib`
     - `seaborn`
     - `joblib`
     - `rasterio`
     - `shapely`

   You can install the required dependencies using:
   ```bash
   pip install -r requirements.txt
   ```

2. **Data Preparation**
   - We will provide the data files required for the project. Please place them in the appropriate directory (`file1.csv` and other files as required).

3. **Running the Project**
   - Run the main program with the following command:
   ```bash
   python main.py
   ```

   This will execute the following tasks:
   - Data loading and preprocessing.
   - Feature selection.
   - Model training.
   - Stacking ensemble.
   - SHAP analysis and feature importance visualization.
   - Correlation heatmap plotting.

## Results and Output

- The best trained model will be saved as a `.joblib` file.
- SHAP analysis results will be displayed showing the feature importance.
- The stacked ensemble model's performance (e.g., RMSE, R²) will be output to the console.
- Maps and other visualizations will be saved as image files.

## Data Availability

The data required for this project (such as `file1.csv` and others) will be provided as per your request. Please contact the project maintainers to access the datasets.

## Contributions

Feel free to raise any issues or suggest improvements! If you have any questions, please don’t hesitate to open an issue or submit a pull request.

---

Thank you for using this project! We hope it helps you achieve better results in data analysis, model training, and interpretability.
