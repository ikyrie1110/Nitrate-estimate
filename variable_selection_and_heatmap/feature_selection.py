import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.preprocessing import MinMaxScaler

def select_features_by_correlation(data, target='Nitrate', threshold=0.3):
    """
    使用相关性矩阵筛选与目标变量相关性较高的特征
    """
    corr_matrix = data.corr()
    corr_with_target = corr_matrix[target].drop(target)
    selected = corr_with_target[abs(corr_with_target) > threshold].index.tolist()
    return selected + [target] if target not in selected else selected

def select_features_by_rf(data, target='Nitrate', threshold=0.05):
    """
    使用随机森林特征重要性筛选特征，并绘图展示
    """
    X = data.drop(columns=[target], errors='ignore')
    y = data[target]
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X, y)
    importances = model.feature_importances_
    
    # 可视化特征重要性
    plt.figure(figsize=(10, 6))
    plt.barh(X.columns, importances)
    plt.xlabel("Feature Importance")
    plt.title("Random Forest Feature Importance")
    plt.tight_layout()
    plt.show()

    selected = X.columns[importances > threshold].tolist()
    return selected + [target] if target not in selected else selected

def evaluate_feature_sets(data, target='Nitrate', feature_sets={}):
    """
    对多个特征集分别训练模型并返回最优特征集
    """
    from model_training import train_and_evaluate
    scores = {}
    best_r2 = -1
    best_name = ''
    best_features = []

    for name, features in feature_sets.items():
        subset = data[features].dropna()
        X = subset.drop(columns=[target], errors='ignore')
        y = subset[target]
        r2 = train_and_evaluate(X, y)
        scores[name] = r2
        if r2 > best_r2:
            best_r2 = r2
            best_name = name
            best_features = features

    return best_name, best_features, scores
