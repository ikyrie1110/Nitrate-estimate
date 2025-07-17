from sklearn.model_selection import KFold
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score

def generate_stacking_features(X, y, models, n_splits=5):
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    meta_features = pd.DataFrame(index=X.index)
    for name, model in models.items():
        preds = np.zeros(len(X))
        for train_idx, val_idx in kf.split(X):
            X_train_fold, X_val_fold = X.iloc[train_idx], X.iloc[val_idx]
            y_train_fold = y.iloc[train_idx]
            model.fit(X_train_fold, y_train_fold)
            preds[val_idx] = model.predict(X_val_fold)
        meta_features[name] = preds
    return meta_features

def stack_and_evaluate(X_train_stack, y_train, X_test_stack, y_test, meta_model):
    meta_model.fit(X_train_stack, y_train)
    y_pred = meta_model.predict(X_test_stack)
    rmse = mean_squared_error(y_test, y_pred, squared=False)
    r2 = r2_score(y_test, y_pred)
    return rmse, r2
