import shap
import matplotlib.pyplot as plt

def shap_analysis(model, X_train, X_test):
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test)

    shap.summary_plot(shap_values, X_test)
    plt.show()
