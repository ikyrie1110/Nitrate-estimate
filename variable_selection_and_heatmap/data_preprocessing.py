import numpy as np
import pandas as pd

def load_data(data_path, importance_path):
    """
    加载数据和特征重要性文件
    """
    data = pd.read_csv(data_path)
    feature_importance_df = pd.read_csv(importance_path)
    return data, feature_importance_df

def preprocess_data(data):
    """
    数据预处理：删除包含缺失值或无穷大值的行
    """
    data = data.replace([np.inf, -np.inf], np.nan)
    data = data.dropna()
    return data
