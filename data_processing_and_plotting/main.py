from data_preprocessing import load_data, preprocess_data
from model_training import prepare_features_and_target, split_data, train_models
from stacking import generate_stacking_features, stack_and_evaluate
from shap_analysis import shap_analysis
from sklearn.linear_model import Ridge

def main():
    # 第一步：数据加载和预处理
    data = load_data('file1.csv')  # 加载数据
    data = preprocess_data(data)   # 进行数据预处理

    # 第二步：特征和目标变量准备
    X, y = prepare_features_and_target(data)  # 提取特征和目标变量

    # 第三步：数据划分
    X_train, X_test, y_train, y_test = split_data(X, y)  # 分割数据集为训练集和测试集

    # 第四步：训练模型
    best_model = train_models(X_train, y_train)  # 使用训练集训练最佳模型

    # 第五步：模型堆叠特征生成
    models = {'RandomForest': best_model}  # 你可以在此处添加更多模型进行堆叠
    X_train_stacking = generate_stacking_features(X_train, y_train, models)  # 生成训练集堆叠特征
    X_test_stacking = generate_stacking_features(X_test, y_test, models)    # 生成测试集堆叠特征

    # 第六步：堆叠集成模型训练与评估
    meta_model = Ridge()  # 可以替换为其他元模型
    rmse, r2 = stack_and_evaluate(X_train_stacking, y_train, X_test_stacking, y_test, meta_model)

    # 打印堆叠集成模型的评估结果
    print(f"Stacked Model RMSE: {rmse:.4f}, R²: {r2:.4f}")

    # 第七步：SHAP分析
    shap_analysis(best_model, X_train, X_test)  # 对最佳模型进行SHAP分析

if __name__ == "__main__":
    main()
