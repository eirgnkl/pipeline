import scanpy as sc
import numpy as np
import squidpy as sq
from sklearn.decomposition import TruncatedSVD

def process(adata_rna, adata_msi, output_rna, output_msi, params=None):
    params = params or {}

    n_components = params.get("n_components", 16)
    n_neighbors = params.get("n_neighbors", 6)

    #Apply SVD on seqRNA RNA
    svd_reducer = TruncatedSVD(n_components=n_components)
    svd_features = svd_reducer.fit_transform(adata_rna.X.toarray())
    adata_rna.obsm["svd_features"] = svd_features

    #graph
    sq.gr.spatial_neighbors(adata_rna, coord_type="grid", spatial_key="spatial", n_neighs=n_neighbors)
    graph_svd = svd_reducer.fit_transform(adata_rna.obsp["spatial_connectivities"])
    adata_rna.obsm["svd_graph_features"] = graph_svd

    #Save 
    adata_rna.write(output_rna)
    adata_msi.write(output_msi)
