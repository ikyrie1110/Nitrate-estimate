import warnings
warnings.filterwarnings("ignore")

from data_preprocessing import load_and_prepare_data
from model_training import train_base_models
from stacking import train_stacking_model
from shap_analysis import shap_analysis_base_models, shap_analysis_stacking
from plotting import plot_model_results, plot_geotiff_with_shapes

# ======================= 1. 加载与预处理数据 =======================
data_path = "final_merged_with_data.csv"
X_train, X_test, y_train, y_test, X_train_full, X_test_full, X_all, y_all, stratify_labels = load_and_prepare_data(data_path)

# ======================= 2. 训练基模型并返回模型集合 =======================
final_models, test_metrics = train_base_models(X_train, X_test, y_train, y_test)

# ======================= 3. 堆叠集成模型训练（返回元模型）=======================
stack_models, meta_model, X_train_stack, X_test_stack = train_stacking_model(
    final_models, X_train_full, X_test_full, y_train, y_test
)

# ======================= 4. SHAP 分析（基模型 + 元模型）=======================
print("\n🔍 开始基模型 SHAP 分析...")
shap_analysis_base_models(final_models, X_train, X_test, X_all)

print("\n🔍 开始堆叠模型 SHAP 分析...")
shap_analysis_stacking("ElasticNet_best_meta_model.joblib", X_train_stack, X_test_stack, stack_models)

# ======================= 5. 空间图绘制（如需绘图）=======================
# 设置你的GeoTIFF文件夹路径 & 输出图像路径
geotiff_folder = "geotiff_maps/"
output_folder = "map_outputs/"
print("\n🗺️ 开始绘制 GeoTIFF 地图...")
plot_geotiff_with_shapes(geotiff_folder, output_folder)

# ======================= 6. 误差比较图（如有）=======================
plot_model_results(test_metrics)

print("\n✅ 全部流程已完成。")
