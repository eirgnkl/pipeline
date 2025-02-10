import numpy as np
import pandas as pd
import scanpy as sc
from sklearn.linear_model import Ridge
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import root_mean_squared_error, r2_score



#pass each parameter seperately for the actual function
def run_ridge_reg(
        adata_rna,
        adata_metabolomics, 
        params, 
        featsel,
        **kwargs):
    
    #adding feature selection as a param to select correct parts of the adata
    if featsel == "hvg":
        X_rna = adata_rna.X  # Use raw HVG-selected features
    elif featsel == "hvg_svd":
        X_rna = adata_rna.obsm["svd_features"]
    elif featsel == "hvg_svd_graph":
        X_rna = adata_rna.obsm["svd_graph_features"]
    elif featsel == "svd":
        X_rna = adata_rna.obsm["svd_features"]
    elif featsel == "svd_graph":
        X_rna = adata_rna.obsm["svd_graph_features"]
    else:
        raise ValueError(f"Unsupported feature selection method: {featsel}")
        
    #msi always processed with just hvd or nothing
    Y_metabolomics = adata_metabolomics.X

    #Train-test split based on 'split' column
    split = adata_rna.obs['split']
    train_idx = np.where(split == 'train')[0]
    test_idx = np.where(split == 'test')[0]

    X_train, X_test = X_rna[train_idx], X_rna[test_idx]
    Y_train, Y_test = Y_metabolomics[train_idx], Y_metabolomics[test_idx]

    # Ridge regression with specified alpha
    alpha = float(params['alpha'] ) # have as function parameter
    ridge = Ridge(alpha=alpha)
    ridge.fit(X_train, Y_train)

    # Predictions and evaluation
    Y_pred = ridge.predict(X_test)

    #Pearson spearman
    pearson_corr = pearsonr(Y_pred.flatten(), Y_test.flatten())[0]
    spearman_corr = spearmanr(Y_pred.flatten(), Y_test.flatten())[0]

    #MSE and R2
    # mse_train = root_mean_squared_error(Y_train, Y_train_pred)
    mse_test = root_mean_squared_error(Y_test, Y_pred)
    # r2_train = r2_score(matching_msi_train, msi_train_pred)
    r2_test = r2_score(Y_test, Y_pred)


    # Save results to a DataFrame
    results = pd.DataFrame({
    'mse': [mse_test],
    'r2': [r2_test],
    'pearson': [pearson_corr],
    'spearman': [spearman_corr], 
    'alpha': [alpha]})

    return results