import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, root_mean_squared_error
from scipy.stats import spearmanr, pearsonr
import numpy as np
from scipy.sparse import issparse

def convert_to_dense(matrix):
    """Converts a sparse matrix to dense if necessary."""
    if issparse(matrix):
        return matrix.toarray()
    return matrix

def run_linreg(
        adata_rna,
        adata_metabolomics,
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

    #convert to dense if needed
    X_rna = convert_to_dense(X_rna)
    Y_metabolomics = convert_to_dense(Y_metabolomics)

    #Train-test split based on 'split' column
    split = adata_rna.obs['split']
    train_idx = np.where(split == 'train')[0]
    test_idx = np.where(split == 'test')[0]

    X_train, X_test = X_rna[train_idx], X_rna[test_idx]
    Y_train, Y_test = Y_metabolomics[train_idx], Y_metabolomics[test_idx]

    # Fit linear regression
    lin = LinearRegression()
    lin.fit(X_train, Y_train)

    # Predictions and evaluation
    Y_pred = lin.predict(X_test)
    
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
    'spearman': [spearman_corr]})

    return results