import scanpy as sc
import numpy as np
from sklearn.decomposition import TruncatedSVD

def process(adata_rna, adata_msi, output_rna, output_msi, params=None):
    params = params or {}

    top_genes = params.get("top_genes", 5000)
    n_components = params.get("n_components", 16)

    #seqRNA
    #apply hvg
    if "highly_variable" not in adata_rna.var.columns:
        sc.pp.highly_variable_genes(adata_rna, flavor='seurat', n_top_genes=top_genes)
    hvg_rna = adata_rna[:, adata_rna.var["highly_variable"]].copy()

    #apply SVD
    svd_reducer = TruncatedSVD(n_components=n_components)
    svd_features = svd_reducer.fit_transform(hvg_rna.X.toarray())
    hvg_rna.obsm["svd_features"] = svd_features


    #MSI processed only for highly variable metabolites, kept hvg_ for uniformality in vars
    if "highly_variable" not in adata_msi.var.columns:
        sc.pp.highly_variable_genes(adata_msi, flavor='seurat', n_top_genes=top_genes)
    hvg_msi = adata_msi[:, adata_msi.var["highly_variable"]].copy()

    #Save
    hvg_rna.write(output_rna)
    hvg_msi.write(output_msi)
