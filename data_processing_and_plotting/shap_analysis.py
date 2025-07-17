import shap
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.ticker as ticker
import joblib
import os

plt.rcParams['font.family'] = 'Times New Roman'

def shap_analysis_base_models(final_models, X_train, X_test, X, save_dir="shap_results"):
    os.makedirs(save_dir, exist_ok=True)
    feature_names = X.columns.tolist()

    for model_name, model in final_models.items():
        print(f"\n===== {model_name} SHAP分析 =====")
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_test)

        fig, ax1 = plt.subplots(figsize=(10, 8))
        shap.summary_plot(shap_values, X_test, feature_names=feature_names, plot_type="dot", show=False, color_bar=True)
        ax1.set_position([0.3, 0.2, 0.6, 0.65])
        ax1 = plt.gca()
        cbar = plt.gcf().axes[-1]
        cbar.set_ylabel('Covariate value', fontsize=20)
        cbar.tick_params(labelsize=20)

        ax2 = ax1.twiny()
        shap.summary_plot(shap_values, X_test, plot_type="bar", show=False)
        ax2.set_position([0.3, 0.2, 0.6, 0.65])

        for bar in ax2.patches:
            bar.set_alpha(0.2)

        ax1.set_xlabel('SHAP value ', fontsize=22)
        ax1.tick_params(axis='y', labelsize=18)
        ax1.tick_params(axis='x', labelsize=20)
        ax2.set_xlabel('Mean(|SHAP value|)', fontsize=22)
        ax2.xaxis.set_label_position('top')
        ax2.xaxis.tick_top()
        ax2.tick_params(axis='x', labelsize=20)
        ax1.xaxis.set_major_formatter(ticker.FormatStrFormatter('%.1f'))
        ax2.xaxis.set_major_formatter(ticker.FormatStrFormatter('%.1f'))
        ax1.set_title(f"{model_name}", fontsize=25, pad=20)
        plt.tight_layout()
        plt.subplots_adjust(left=0.1, right=0.95, top=0.85, bottom=0.1)

        plt.savefig(os.path.join(save_dir, f"{model_name}_SHAP_combined.png"), format='png', bbox_inches='tight', dpi=880)
        plt.show()

def shap_analysis_stacking(best_meta_model_path, X_train_stack, X_test_stack, stack_models, save_path="stacking_SHAP_combined.png"):
    best_meta_model = joblib.load(best_meta_model_path)
    meta_explainer = shap.LinearExplainer(best_meta_model, X_train_stack)
    meta_shap_values = meta_explainer.shap_values(X_test_stack)

    base_model_names = list(stack_models.keys())
    n_base_models = len(base_model_names)
    base_shap = meta_shap_values[:, :n_base_models]
    el_weights = best_meta_model.coef_[:n_base_models]
    shap_means = np.mean(np.abs(base_shap), axis=0)

    corr_coef = np.corrcoef(el_weights, shap_means)[0, 1]
    print(f"ElasticNet权重与SHAP值平均的相关系数: {corr_coef:.4f}")

    fig, ax1 = plt.subplots(figsize=(8, 8), dpi=300)
    plt.rcParams['font.family'] = 'Times New Roman'
    shap.summary_plot(meta_shap_values[:, :n_base_models], X_test_stack[:, :n_base_models],
                      feature_names=base_model_names, plot_type="dot", show=False, color_bar=True)
    ax1.set_position([0.2, 0.2, 0.6, 0.65])
    ax1 = plt.gca()
    cbar = plt.gcf().axes[-1]
    cbar.set_ylabel('Covariate value', fontsize=20)
    cbar.tick_params(labelsize=20)

    ax2 = ax1.twiny()
    shap.summary_plot(meta_shap_values[:, :n_base_models], X_test_stack[:, :n_base_models],
                      feature_names=base_model_names, plot_type="bar", show=False)
    ax2.set_position([0.2, 0.2, 0.6, 0.65])
    for bar in ax2.patches:
        bar.set_alpha(0.3)

    ax1.set_xlabel('SHAP value ', fontsize=22)
    ax2.set_xlabel('Mean(|SHAP value|)', fontsize=22)
    ax2.xaxis.set_label_position('top')
    ax2.xaxis.tick_top()
    ax1.tick_params(axis='y', labelsize=18)
    ax1.tick_params(axis='x', labelsize=20)
    ax2.tick_params(axis='x', labelsize=20)
    ax1.xaxis.set_major_formatter(ticker.FormatStrFormatter('%.1f'))
    ax2.xaxis.set_major_formatter(ticker.FormatStrFormatter('%.1f'))
    ax1.set_title("Stacking Model SHAP Analysis", fontsize=25, pad=20)
    plt.tight_layout()
    plt.subplots_adjust(left=0.2, right=0.95, top=0.65, bottom=0.2)

    plt.savefig(save_path, format='png', bbox_inches='tight', dpi=880)
    plt.show()

    average_shap_values = np.mean(meta_shap_values, axis=0)
    print("各基模型的平均 SHAP 值：")
    for i, model_name in enumerate(stack_models.keys()):
        print(f"{model_name}: {average_shap_values[i]:.4f}")
