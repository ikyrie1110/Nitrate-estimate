from data_preprocessing import load_data, preprocess_data
from feature_selection import (
    select_features_by_correlation,
    select_features_by_rf,
    evaluate_feature_sets
)
from heatmap import plot_correlation_heatmap

def main():
    # 1. 加载与清洗数据
    data = load_data("final_merged_with_data.csv")
    data = preprocess_data(data)

    # 2. 可视化热力图
    plot_correlation_heatmap(data)

    # 3. 特征选择：相关性 vs 随机森林
    corr_features = select_features_by_correlation(data)
    rf_features = select_features_by_rf(data)
    union_features = list(set(corr_features).union(set(rf_features)))

    feature_sets = {
        "Correlation": corr_features,
        "RandomForest": rf_features,
        "Combined": union_features
    }

    # 4. 评估各特征集性能，输出最佳
    best_name, best_features, score_dict = evaluate_feature_sets(
        data, feature_sets=feature_sets)

    print(f"\n✅ 最佳特征集为: {best_name}")
    print(f"📌 最佳特征列表为: {best_features}")
    print(f"\n📊 各特征集 R² 得分：")
    for name, score in score_dict.items():
        print(f"{name}: R² = {score:.4f}")

if __name__ == "__main__":
    main()
