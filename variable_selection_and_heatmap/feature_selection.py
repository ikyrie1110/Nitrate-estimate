def calculate_feature_importance(feature_importance_df, threshold=0.5):
    """
    计算特征的重要性，并根据阈值筛选特征
    """
    if 'Average_Importance' not in feature_importance_df.columns:
        lasso_columns = [col for col in feature_importance_df.columns if 'Lasso_Station_' in col]
        feature_importance_df['Average_Importance'] = feature_importance_df[lasso_columns].mean(axis=1)
    
    selected_features = feature_importance_df[feature_importance_df['Average_Importance'] > threshold]['Features'].values
    return selected_features
