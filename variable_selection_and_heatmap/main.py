from data_preprocessing import load_data, preprocess_data
from feature_selection import calculate_feature_importance
from heatmap import plot_correlation_heatmap
from model_training import train_station_model

def main():
    # 加载数据
    data, feature_importance_df = load_data('file1.csv', 'file2.csv')

    # 数据预处理
    data = preprocess_data(data)

    # 特征选择
    selected_features = calculate_feature_importance(feature_importance_df)

    # 绘制热力图
    plot_correlation_heatmap(data)

    # 模型训练与评估
    stations = data['STATION_ID'].unique()
    for station in stations:
        train_station_model(station, data, selected_features)

if __name__ == "__main__":
    main()
