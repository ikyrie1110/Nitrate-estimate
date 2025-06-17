from sklearn.model_selection import KFold
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score

def generate_stacking_features(X, y, models, n_splits=5):
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    stacking_features = np.zeros((X.shape[0], len(models)))

    for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
        X_train_fold, X_val_fold = X.iloc[train_idx], X.iloc[val_idx]
        y_train_fold = y.iloc[train_idx]

        for model_idx, (name, model) in enumerate(models.items()):
            model.fit(X_train_fold, y_train_fold)
            stacking_features[val_idx, model_idx] = model.predict(X_val_fold)

    return stacking_features

def stack_and_evaluate(X_train_stacking, y_train, X_test_stacking, y_test, meta_model):
    meta_model.fit(X_train_stacking, y_train)
    y_pred = meta_model.predict(X_test_stacking)

    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)

    return rmse, r2
