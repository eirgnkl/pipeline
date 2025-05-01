import numpy as np
import pandas as pd
import scanpy as sc
from sklearn.linear_model import Ridge
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import root_mean_squared_error, r2_score
from sklearn.metrics import mean_absolute_error



#maybe update pipeline to pass each parameter seperately for the actual function at some point
def run_ridge_reg(
        adata_rna_train,
        adata_rna_test,
        adata_msi_train,
        adata_msi_test, 
        params, 
        featsel,
        **kwargs):
    
    if featsel in {"hvg", "hvg_nomsi", "none"}:
        X_train = adata_rna_train.X  
        X_test = adata_rna_test.X  
        Y_train, Y_test = adata_msi_train.X, adata_msi_test.X
    elif featsel in {"hvg_svd", "svd", "svd_hvgmsi"}:
        X_train = adata_rna_train.obsm["svd_features"]
        X_test = adata_rna_test.obsm["svd_features"]
        Y_train, Y_test = adata_msi_train.X, adata_msi_test.X
    elif featsel in {"hvg_svd_graph", "svd_graph", "svd_graph_hvgmsi"}:
        X_train = adata_rna_train.obsm["svd_graph"]
        X_test = adata_rna_test.obsm["svd_graph"] 
        Y_train, Y_test = adata_msi_train.X, adata_msi_test.X
    elif featsel == "hvg_rna":
        X_train = adata_rna_train.X  
        X_test = adata_rna_test.X  
        Y_train, Y_test = adata_msi_train.obsm["X_pca_split"], adata_msi_test.obsm["X_pca_split"]
    elif featsel == "hvg_rna_svd":
        X_train = adata_rna_train.obsm["svd_features"]
        X_test = adata_rna_test.obsm["svd_features"]
        Y_train, Y_test = adata_msi_train.obsm["X_pca_split"], adata_msi_test.obsm["X_pca_split"]
    elif featsel == "hvg_rna_svd_graph":
        X_train = adata_rna_train.obsm["svd_graph"]
        X_test = adata_rna_test.obsm["svd_graph"] 
        Y_train, Y_test = adata_msi_train.obsm["X_pca_split"], adata_msi_test.obsm["X_pca_split"]
    else:
        raise ValueError(f"Unsupported feature selection method: {featsel}")


    # Ridge regression with specified alpha
    alpha = float(params['alpha'] ) # have as function parameter
    ridge = Ridge(alpha=alpha)
    ridge.fit(X_train, Y_train)

    # Predictions and evaluation
    Y_pred = ridge.predict(X_test)
    Y_pred_train = ridge.predict(X_train)

    # Compute training metrics
    rmse_train = root_mean_squared_error(Y_train, Y_pred_train)
    r2_train = r2_score(Y_train, Y_pred_train)
    mae_train = mean_absolute_error(Y_train, Y_pred_train)

    #Pearson spearman
    pearson_corr = pearsonr(Y_pred.flatten(), Y_test.flatten())[0]
    spearman_corr = spearmanr(Y_pred.flatten(), Y_test.flatten())[0]

    #MSE and R2
    # mse_train = root_mean_squared_error(Y_train, Y_train_pred)
    rmse_test = root_mean_squared_error(Y_test, Y_pred)
    # r2_train = r2_score(matching_msi_train, msi_train_pred)
    r2_test = r2_score(Y_test, Y_pred)
    mae_test = mean_absolute_error(Y_test, Y_pred)

    # Compute per-metabolite (column-wise) Pearson and Spearman
    per_met_pearsons = [pearsonr(Y_pred[:, i], Y_test[:, i])[0] for i in range(Y_test.shape[1])]
    per_met_spearmans = [spearmanr(Y_pred[:, i], Y_test[:, i])[0] for i in range(Y_test.shape[1])]

    avg_pearson_per_metabolite = np.nanmean(per_met_pearsons)
    avg_spearman_per_metabolite = np.nanmean(per_met_spearmans)
    
    # Per-metabolite RMSE and relative RMSE
    per_met_rmse = [np.sqrt(np.mean((Y_pred[:, i] - Y_test[:, i]) ** 2)) for i in range(Y_test.shape[1])]
    per_met_mean = [np.mean(Y_test[:, i]) for i in range(Y_test.shape[1])]
    rel_rmse = [r / m if m != 0 else np.nan for r, m in zip(per_met_rmse, per_met_mean)]
    avg_rel_rmse = np.nanmean(rel_rmse)

    # Save results to a DataFrame
    metrics = pd.DataFrame({
        'rmse': [rmse_test],
        'mae': [mae_test],
        'r2': [r2_test],
        'pearson': [pearson_corr],
        'spearman': [spearman_corr],
        'avg_pearson_per_metabolite': [avg_pearson_per_metabolite],
        'avg_spearman_per_metabolite': [avg_spearman_per_metabolite],
        'avg_rel_rmse': [avg_rel_rmse],
        'train_rmse': [rmse_train],
        'train_mae': [mae_train],
        'train_r2': [r2_train]
    })


    #Add this for interpretability later, check outputs of each model's preds
    predictions = pd.DataFrame({
        'y_true': Y_test.flatten(),
        'y_pred': Y_pred.flatten()
    })  
    return metrics, predictions
