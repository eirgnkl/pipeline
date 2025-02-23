import numpy as np
import pandas as pd
import scanpy as sc
from sklearn.linear_model import ElasticNet
from sklearn.metrics import root_mean_squared_error, r2_score
from scipy.stats import pearsonr, spearmanr

def run_elastic_net(
        adata_rna_train,
        adata_rna_test,
        adata_msi_train,
        adata_msi_test,
        params, 
        featsel,
        **kwargs):
    # Select features based on the provided feature selection method
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

    # Retrieve hyperparameters from the params dictionary
    alpha = float(params['alpha'])
    l1_ratio = float(params['l1_ratio'])

    # Initialize and train the ElasticNet model
    elastic_net = ElasticNet(alpha=alpha, l1_ratio=l1_ratio)
    elastic_net.fit(X_train, Y_train)

    # Make predictions on the test set
    Y_pred = elastic_net.predict(X_test)

    # Compute evaluation metrics
    pearson_corr = pearsonr(Y_pred.flatten(), Y_test.flatten())[0]
    spearman_corr = spearmanr(Y_pred.flatten(), Y_test.flatten())[0]
    rmse_test = root_mean_squared_error(Y_test, Y_pred)
    r2_test = r2_score(Y_test, Y_pred)

    # Save the evaluation results to a DataFrame
    results = pd.DataFrame({
        'rmse': [rmse_test],
        'r2': [r2_test],
        'pearson': [pearson_corr],
        'spearman': [spearman_corr]
    })

    return results
