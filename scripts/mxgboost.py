import numpy as np
import pandas as pd
import scanpy as sc
from xgboost import XGBRegressor
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import root_mean_squared_error, r2_score


# Function to run XGBoost regression
def run_xgboost(
        adata_rna_train,
        adata_rna_test,
        adata_msi_train,
        adata_msi_test, 
        params, 
        featsel,
        **kwargs):
    
    #adding feature selection as a param to select correct parts of the adata
    if featsel == "hvg":
        X_train = adata_rna_train.X  
        X_test = adata_rna_test.X  
        Y_train, Y_test = adata_msi_train.X, adata_msi_test.X
    elif featsel == "hvg_svd":
        X_train = adata_rna_train.obsm["svd_features"]
        X_test = adata_rna_test.obsm["svd_features"]
        Y_train, Y_test = adata_msi_train.X, adata_msi_test.X
    elif featsel == "hvg_svd_graph":
        X_train = adata_rna_train.obsm["svd_graph"]
        X_test = adata_rna_test.obsm["svd_graph"] 
        Y_train, Y_test = adata_msi_train.X, adata_msi_test.X
    elif featsel == "svd":
        X_train = adata_rna_train.obsm["svd_features"]
        X_test = adata_rna_test.obsm["svd_features"]
        Y_train, Y_test = adata_msi_train.X, adata_msi_test.X
    elif featsel == "svd_graph":
        X_train = adata_rna_train.obsm["svd_graph"]
        X_test = adata_rna_test.obsm["svd_graph"]
        Y_train, Y_test = adata_msi_train.X, adata_msi_test.X
    else:
        raise ValueError(f"Unsupported feature selection method: {featsel}")



    # XGBoost Hyperparameters
    alpha = float(params.get("alpha", 50))  # L1 regularization (Lasso)
    lambda_ = float(params.get("lambda", 100))  # L2 regularization (Ridge)
    max_depth = int(params.get("max_depth", 3))
    learning_rate = float(params.get("learning_rate", 0.05))
    n_estimators = int(params.get("n_estimators", 1000))
    subsample = float(params.get("subsample", 0.8))
    colsample_bytree = float(params.get("colsample_bytree", 0.8))
    min_child_weight = int(params.get("min_child_weight", 3))
    early_stopping_rounds = int(params.get("early_stopping_rounds", 10))

    # Initialize XGBoost model
    xgb_model = XGBRegressor(
        reg_alpha=alpha, 
        reg_lambda=lambda_,
        max_depth=max_depth,
        learning_rate=learning_rate,
        n_estimators=n_estimators,
        subsample=subsample,
        colsample_bytree=colsample_bytree,
        min_child_weight=min_child_weight,
        early_stopping_rounds=early_stopping_rounds,
        n_jobs=-1
    )

    # Train the model
    xgb_model.fit(
        X_train, 
        Y_train, 
        eval_set=[(X_test, Y_test)], 
        verbose=False
    )

    # Predictions and evaluation
    Y_pred = xgb_model.predict(X_test)

    # Pearson and Spearman correlation
    pearson_corr = pearsonr(Y_pred.flatten(), Y_test.flatten())[0]
    spearman_corr = spearmanr(Y_pred.flatten(), Y_test.flatten())[0]

    #MSE and R2
    mse_test = root_mean_squared_error(Y_test, Y_pred)
    r2_test = r2_score(Y_test, Y_pred)

    # Save results to a DataFrame
    results = pd.DataFrame({
        "mse": [mse_test],
        "r2": [r2_test],
        "pearson": [pearson_corr],
        "spearman": [spearman_corr],
        "alpha": [alpha],
        "lambda": [lambda_],
        "max_depth": [max_depth],
        "learning_rate": [learning_rate],
        "n_estimators": [n_estimators],
        "subsample": [subsample],
        "colsample_bytree": [colsample_bytree],
        "min_child_weight": [min_child_weight]
    })

    return results
