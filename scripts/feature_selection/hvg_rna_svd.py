import scanpy as sc
import numpy as np
from sklearn.decomposition import TruncatedSVD

def process(adata_rna, adata_msi, output_rna_train, output_rna_test, output_msi_train, output_msi_test, split, params=None):
   
   #Extract input params
    params = params or {}
    top_genes = params.get("top_genes", 5000)
    n_components = params.get("n_components", 100)
    split_name = split

    #----------------------------------------------sc-seqRNA----------------------------------------------#
    #-----HVG-----#
    if "highly_variable" not in adata_rna.var.columns:
        sc.pp.highly_variable_genes(adata_rna, flavor='seurat', n_top_genes=top_genes)
    hvg_rna = adata_rna[:, adata_rna.var["highly_variable"]].copy()
    
    #split train and test, used for svd and rest
    hvg_rna_train = hvg_rna[hvg_rna.obs[split_name] == "train"].copy()
    hvg_rna_test = hvg_rna[hvg_rna.obs[split_name] == "test"].copy()

    #-----SVD-----#
    svd_reducer = TruncatedSVD(n_components=n_components) 

    svd_features_train = svd_reducer.fit_transform(hvg_rna_train.X.toarray())
    hvg_rna_train.obsm["svd_features"] = svd_features_train

    svd_features_test = svd_reducer.fit_transform(hvg_rna_test.X.toarray())
    hvg_rna_test.obsm["svd_features"] = svd_features_test

    #----------------------------------------------MSI----------------------------------------------#
    #-----no preprocessing of msi at all-----#
    msi_train = adata_msi[adata_msi.obs[split_name] == "train"].copy()
    msi_test = adata_msi[adata_msi.obs[split_name] == "test"].copy()

    #----------------------------------------------SAVE----------------------------------------------#
    hvg_rna_train.write(output_rna_train)
    hvg_rna_test.write(output_rna_test)
    msi_train.write(output_msi_train)
    msi_test.write(output_msi_test)

