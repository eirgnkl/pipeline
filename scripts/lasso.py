import scanpy as sc
import pandas as pd
from sklearn.linear_model import Lasso
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, root_mean_squared_error
from sklearn.metrics import mean_absolute_error

from scipy.stats import spearmanr, pearsonr
import numpy as np

def run_lasso(
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



    # Lasso regression with specified alpha
    alpha = float(params['alpha'] ) # have as function parameter
    lassi = Lasso(alpha=alpha)
    lassi.fit(X_train, Y_train)

    # Predictions and evaluation
    Y_pred = lassi.predict(X_test)
    
    #Pearson spearman
    pearson_corr = pearsonr(Y_pred.flatten(), Y_test.flatten())[0]
    spearman_corr = spearmanr(Y_pred.flatten(), Y_test.flatten())[0]

    #MSE and R2
    # mse_train = root_mean_squared_error(Y_train, Y_train_pred)
    rmse_test = root_mean_squared_error(Y_test, Y_pred)
    # r2_train = r2_score(matching_msi_train, msi_train_pred)
    r2_test = r2_score(Y_test, Y_pred)
    mae_test = mean_absolute_error(Y_test, Y_pred)

    #Save results to a DataFrame
    metrics = pd.DataFrame({
        'rmse': [rmse_test],
        'mae': [mae_test],
        'r2': [r2_test],
        'pearson': [pearson_corr],
        'spearman': [spearman_corr]
    })
    #Add this for interpretability later, check outputs of each model's preds
    predictions = pd.DataFrame({
        'y_true': Y_test.flatten(),
        'y_pred': Y_pred.flatten()
    })

    return metrics, predictions
