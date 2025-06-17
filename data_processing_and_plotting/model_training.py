import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor
from catboost import CatBoostRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

def prepare_features_and_target(data):
    X = data[['Year', 'lon', 'NH₄⁺', 'lat', 'SO₄²⁻',......]]
    y = data['Nitrate']
    return X, y

# 数据划分
def split_data(X, y):
    # 按季度进行分层抽样，并划分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    return X_train, X_test, y_train, y_test

# 模型训练与调优
def train_models(X_train, y_train):
    # 定义模型及其参数网格
    models = {
        'RandomForest': (RandomForestRegressor(random_state=42), {
            'n_estimators': [50, 60, 70],
            'max_depth': [5, 6, 7],
            'min_samples_split': [8, 10],
            'min_samples_leaf': [6, 8],
            'max_features': ['sqrt']
        }),
        'GradientBoosting': (GradientBoostingRegressor(random_state=42), {
            'n_estimators': [300, 500],
            'max_depth': [3, 4, 5],
            'learning_rate': [0.01, 0.02, 0.03],
            'min_samples_split': [10, 15, 20],
            'min_samples_leaf': [3, 5, 8],
            'max_features': [0.6, 0.7, 0.8],
            'subsample': [0.6, 0.7, 0.8]
        }),
        'XGBoost': (XGBRegressor(random_state=42, n_jobs=1), {
            'n_estimators': [300, 400],
            'learning_rate': [0.01, 0.02],
            'max_depth': [4, 5],
            'subsample': [0.7, 0.8],
            'reg_alpha': [0.5, 1],
            'reg_lambda': [1, 2]
        }),
        'CatBoost': (CatBoostRegressor(verbose=0, random_state=42), {
            'iterations': [500, 600],
            'depth': [5, 6],
            'learning_rate': [0.02, 0.03],
            'l2_leaf_reg': [5, 6]
        })
    }

    # 初始化记录最佳模型的变量
    best_model = None
    best_rmse = float('inf')
    
    # 设置交叉验证策略
    kfold = KFold(n_splits=10, shuffle=True, random_state=42)

    # 遍历每个模型和其参数网格，进行超参数调优
    for model_name, (model, param_grid) in models.items():
        print(f"\n训练模型: {model_name}")

        # 使用网格搜索进行超参数调优
        grid_search = GridSearchCV(
            estimator=model,
            param_grid=param_grid,
            cv=kfold,
            scoring='neg_mean_squared_error',
            n_jobs=-1,
            verbose=1
        )
        grid_search.fit(X_train, y_train)

        # 输出最佳参数
        print(f"最佳参数：{grid_search.best_params_}")

        # 使用最佳参数初始化模型
        best_model_for_this = grid_search.best_estimator_

        # 在训练集上进行预测并计算RMSE
        y_train_pred = best_model_for_this.predict(X_train)
        rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))

        # 记录当前最佳模型
        if rmse < best_rmse:
            best_rmse = rmse
            best_model = best_model_for_this

        # 打印每个模型的调优结果
        print(f"{model_name} - Best Params: {grid_search.best_params_}, Best RMSE: {rmse:.4f}")

    return best_model

# 评估模型性能
def evaluate_model(best_model, X_test, y_test):
    # 在测试集上进行预测
    y_test_pred = best_model.predict(X_test)

    # 计算RMSE、MAE、R²等评估指标
    test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
    test_mae = mean_absolute_error(y_test, y_test_pred)
    test_r2 = r2_score(y_test, y_test_pred)

    print(f"测试集 RMSE: {test_rmse:.4f}")
    print(f"测试集 MAE: {test_mae:.4f}")
    print(f"测试集 R²: {test_r2:.4f}")

    return test_rmse, test_r2

