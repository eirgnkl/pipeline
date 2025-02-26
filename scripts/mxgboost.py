import numpy as np
import pandas as pd
import scanpy as sc
import cupy as cp  # For GPU array conversion
from xgboost import XGBRegressor
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import root_mean_squared_error, r2_score  # Using MSE then taking sqrt
from scipy.sparse import issparse


def ensure_gpu(data):
    """
    Convert the data to a GPU array if it is not already.
    If the data is a pandas DataFrame or NumPy array, convert it to a CuPy array.
    """
    # If data already has the __cuda_array_interface__, assume it is a GPU array.
    if hasattr(data, '__cuda_array_interface__'):
        return data
    # If data is a DataFrame, convert its underlying NumPy array.
    if isinstance(data, pd.DataFrame):
        return cp.asarray(data.values)
    # If data is a NumPy array, convert it.
    if isinstance(data, np.ndarray):
        return cp.asarray(data)
    # Otherwise, attempt conversion.
    return cp.asarray(data)

def ensure_cpu(data):
    """
    Convert the data to a CPU (NumPy) array if it is on the GPU.
    """
    if hasattr(data, '__cuda_array_interface__'):
        return cp.asnumpy(data)
    return data

# Function to run XGBoost regression
def run_xgboost(
        adata_rna_train,
        adata_rna_test,
        adata_msi_train,
        adata_msi_test, 
        params, 
        featsel,
        **kwargs):
    
    # Select features based on the provided method.
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

    if issparse(X_train):
        X_train = X_train.toarray()
    if issparse(X_test):
        X_test = X_test.toarray()
    if issparse(Y_train):
        Y_train = Y_train.toarray()
    if issparse(Y_test):
        Y_test = Y_test.toarray()

    #Since the model is configured to run on GPU (device="cuda"),
    #convert all input data to GPU arrays if they are not already.
    #If you don't need that, you can just comment out the conversions
    X_train = ensure_gpu(X_train)
    X_test = ensure_gpu(X_test)
    Y_train = ensure_gpu(Y_train)
    Y_test = ensure_gpu(Y_test)

    # XGBoost Hyperparameters
    alpha = float(params.get("alpha", 10))         # L1 regularization (Lasso)
    lambda_ = float(params.get("lambda", 50))       # L2 regularization (Ridge)
    max_depth = int(params.get("max_depth", 5))
    learning_rate = float(params.get("learning_rate", 0.1))
    n_estimators = int(params.get("n_estimators", 500))
    subsample = float(params.get("subsample", 0.9))
    colsample_bytree = float(params.get("colsample_bytree", 0.7))
    min_child_weight = int(params.get("min_child_weight", 2))
    early_stopping_rounds = int(params.get("early_stopping_rounds", 20))
    n_jobs = int(params.get("n_jobs", 15))


    # Initialize XGBoost model on GPU
    xgb_model = XGBRegressor(
        device="cuda",
        reg_alpha=alpha, 
        reg_lambda=lambda_,
        max_depth=max_depth,
        learning_rate=learning_rate,
        n_estimators=n_estimators,
        subsample=subsample,
        colsample_bytree=colsample_bytree,
        min_child_weight=min_child_weight,
        early_stopping_rounds=early_stopping_rounds,
        n_jobs=n_jobs
    )

    # Train the model
    xgb_model.fit(
        X_train, 
        Y_train, 
        eval_set=[(X_test, Y_test)], 
        verbose=False
    )

    # Predict on the test data
    Y_pred = xgb_model.predict(X_test)

    # Convert predictions and test labels back to CPU (NumPy) for evaluation
    Y_pred = ensure_cpu(Y_pred)
    Y_test_cpu = ensure_cpu(Y_test)

    # Pearson and Spearman correlation
    pearson_corr = pearsonr(Y_pred.flatten(), Y_test_cpu.flatten())[0]
    spearman_corr = spearmanr(Y_pred.flatten(), Y_test_cpu.flatten())[0]

    # Compute Root Mean Squared Error and R2 score
    rmse_test = root_mean_squared_error(Y_test_cpu, Y_pred)
    r2_test = r2_score(Y_test_cpu, Y_pred)

    #Save results to a DataFrame
    metrics = pd.DataFrame({
        'rmse': [rmse_test],
        'r2': [r2_test],
        'pearson': [pearson_corr],
        'spearman': [spearman_corr]
    })

    #Add this for interpretability later, check outputs of each model's preds
    predictions = pd.DataFrame({
        'y_true': Y_test_cpu.flatten(),
        'y_pred': Y_pred.flatten()
    })

    return metrics, predictions
# Got this error, had to add conversion to and from GPU arrays for the run of the model
# UserWarning: [14:56:43] WARNING: /home/conda/feedstock_root/build_artifacts/xgboost-split_1738880369036/work/src/common/error_msg.cc:58: 
# Falling back to prediction using DMatrix due to mismatched devices. 
# This might lead to higher memory usage and slower performance. 
# XGBoost is running on: cuda:0, while the input data is on: cpu.
# Potential solutions:
# - Use a data structure that matches the device ordinal in the booster.
# - Set the device for booster before call to inplace_predict.

# This warning will only be shown once.