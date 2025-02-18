import scanpy as sc
import numpy as np
from sklearn.decomposition import TruncatedSVD

def process(adata_rna, adata_msi, output_rna_train, output_rna_test, output_msi_train, output_msi_test, split, params=None):
      
    #Extract input params
    params = params or {}
    n_components = params.get("n_components", 16)
    split_name = split
    
    #Split into train test
    rna_train = adata_rna[adata_rna.obs[split_name] == "train"]
    rna_test = adata_rna[adata_rna.obs[split_name] == "test"]

    #----------------------------------------------sc-seqRNA----------------------------------------------#
    #-----SVD-----#
    svd_reducer = TruncatedSVD(n_components=n_components)

    svd_features_train = svd_reducer.fit_transform(rna_train.X.toarray())
    rna_train.obsm["svd_features"] = svd_features_train

    svd_features_test = svd_reducer.fit_transform(rna_test.X.toarray())
    rna_test.obsm["svd_features"] = svd_features_test

    #----------------------------------------------MSI----------------------------------------------#
    #-----no process-----#
    msi_train = adata_msi[adata_msi.obs[split_name] == "train"]
    msi_test = adata_msi[adata_msi.obs[split_name] == "test"]

    #----------------------------------------------SAVE----------------------------------------------#
    rna_train.write(output_rna_train)
    rna_test.write(output_rna_test)
    msi_train.write(output_msi_train)
    msi_test.write(output_msi_test)

