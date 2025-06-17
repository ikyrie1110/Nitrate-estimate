import numpy as np
from sklearn.linear_model import Lasso
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

def combine_predictions(predictions):
    """
    通过对多个模型的预测结果进行加权平均，得到最终预测结果。
    """
    min_len = min([len(pred) for pred in predictions])
    trimmed_predictions = [pred[:min_len] for pred in predictions]
    return np.mean(trimmed_predictions, axis=0)

def prepare_selected_features(station_id, data, selected_features, forecast_steps):
    """
    根据选定的特征，准备用于模型训练的特征和标签。
    """
    features = pd.DataFrame()
    station_data = data[data['STATION_ID'] == station_id].copy()

    if 'DATE' in station_data.columns:
        dates = pd.to_datetime(station_data['DATE'], errors='coerce')
        station_data = station_data.drop(columns=['DATE'], errors='ignore')

    available_features = station_data.columns
    valid_features = []

    for feature in selected_features:
        if feature in available_features:
            valid_features.append(feature)
        else:
            corrected_feature = feature.replace('_Lag_', '_')
            if corrected_feature in available_features:
                valid_features.append(corrected_feature)
            else:
                print(f"特征 {feature} 在数据中不存在")

    for feature in valid_features:
        features[feature] = station_data[feature]

    valid_idx = features.dropna().index
    features = features.dropna().reset_index(drop=True)
    dates = dates.loc[valid_idx].reset_index(drop=True)

    min_len = len(features) - forecast_steps + 1
    y = station_data['no3-'].shift(-forecast_steps).iloc[:min_len].reset_index(drop=True)

    features = features.iloc[:min_len].reset_index(drop=True)
    dates = dates.iloc[:min_len].reset_index(drop=True)

    return features, y, dates

def rolling_window_split(X, y, history_length, forecast_steps):
    """
    使用滚动窗口方法将数据分割为训练数据和标签。
    """
    X_rolling = []
    y_rolling = []
    for i in range(len(X) - history_length - forecast_steps + 1):
        X_rolling.append(X[i:i + history_length].values)
        y_rolling.append(y[i + history_length:i + history_length + forecast_steps].values)
    return np.array(X_rolling), np.array(y_rolling)

def train_station_model(station, data, selected_features, history_length=5, forecast_steps=1):
    """
    训练单个站点的预测模型
    """
    X, y, dates = prepare_selected_features(station, data, selected_features, forecast_steps)

    X_rolling, y_rolling = rolling_window_split(X, y, history_length, forecast_steps)
    X_reshaped = X_rolling.reshape(X_rolling.shape[0], -1)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_reshaped)

    tscv = TimeSeriesSplit(n_splits=5)
    model = Lasso(alpha=0.1, max_iter=10000)
    predictions = []

    for train_index, test_index in tscv.split(X_scaled):
        X_train, X_test = X_scaled[train_index], X_scaled[test_index]
        y_train, y_test = y_rolling[train_index], y_rolling[test_index]

        model.fit(X_train, y_train.ravel())
        pred = model.predict(X_test)
        predictions.append(pred)

    final_predictions = combine_predictions(predictions)

    rmse_test = np.sqrt(mean_squared_error(y_rolling[:len(final_predictions)], final_predictions))
    r2_test = r2_score(y_rolling[:len(final_predictions)], final_predictions)
    mae_test = mean_absolute_error(y_rolling[:len(final_predictions)], final_predictions)

    print(f"Station {station} - Test RMSE: {rmse_test:.4f}, Test R²: {r2_test:.4f}, Test MAE: {mae_test:.4f}")
