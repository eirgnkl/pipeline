from xgboost import XGBRegressor


def run_xgboost(
        adata_rna,
        adata_metabolomics, 
        params,
        featsel,
        **kwargs):
    
    xgb_model = XGBRegressor(device="cuda", n_jobs=20, reg_alpha=50, \
                        reg_lambda=100, max_depth=3, subsample=0.8, colsample_bytree=0.8, \
                        learning_rate=0.05, n_estimators=1000, min_child_weight=3, early_stopping_rounds=10)

    # Train
    xgb_model.fit(adata_rna, adata_msi, eval_set=[(adata_rna, adata_metabolomics)], verbose=True)

    # Predict
    predictions = xgb_model.predict(adata_rna_feat_test)


    return (predictions)

import numpy as np
import pandas as pd
import scanpy as sc
from xgboost import XGBRegressor
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import root_mean_squared_error, r2_score


# Function to run XGBoost regression
def run_xgboost(adata_rna, 
                    adata_metabolomics,
                    params, 
                    **kwargs):

    # Extract data matrices
    split = adata_rna.obs['split']
    X_rna = adata_rna.X
    Y_metabolomics = adata_metabolomics.X

    # Train-test split based on 'split' column
    train_idx = np.where(split == 'train')[0]
    test_idx = np.where(split == 'test')[0]

    X_train, X_test = X_rna[train_idx], X_rna[test_idx]
    Y_train, Y_test = Y_metabolomics[train_idx], Y_metabolomics[test_idx]

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
    mse_test = root_mean_squared_error(Y_test, Y_pred, squared=False)
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
