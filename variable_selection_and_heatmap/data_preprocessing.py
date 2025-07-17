import pandas as pd
import numpy as np

def load_data(file_path):
    """
    加载 CSV 文件为 DataFrame
    """
    return pd.read_csv(file_path)

def preprocess_data(data):
    """
    清洗数据：转换为数值型，处理缺失值
    """
    data = data.drop(columns=[
        'station', 'GDP', 'X_Lon', 'Y_Lat', 'station_lon', 'station_lat',
        'GridID', "Interpolation_Status", "DATE", "Year_Month"
    ], errors='ignore')
    data = data.apply(pd.to_numeric, errors='coerce')
    return data.fillna(data.mean())
