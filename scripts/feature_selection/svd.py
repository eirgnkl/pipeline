import scanpy as sc
import numpy as np
from sklearn.decomposition import TruncatedSVD

def process(adata_rna, adata_msi, output_rna, output_msi, params=None):
    params = params or {}

    n_components = params.get("n_components", 16)
    #Apply SVD on seqRNA RNA
    svd_reducer = TruncatedSVD(n_components=n_components)
    
    svd_features = svd_reducer.fit_transform(adata_rna.X.toarray())
    adata_rna.obsm["svd_features"] = svd_features

    #Save
    adata_rna.write(output_rna)
    adata_msi.write(output_msi)
