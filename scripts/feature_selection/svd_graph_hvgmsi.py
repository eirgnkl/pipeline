import scanpy as sc
import numpy as np
import squidpy as sq
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import StandardScaler
import pandas as pd


def process(adata_rna, adata_msi, output_rna_train, output_rna_test, output_msi_train, output_msi_test, split, params=None):
    # Extract input params
    
    params = params or {}
    top_mets = params.get("top_mets", 500)
    n_components = params.get("n_components", 128)
    n_components_graph = params.get("n_components_graph", 32)
    n_neighbors = params.get("n_neighbors", 6)
    split_name = split

    # Ensure unique names (preserves order)
    adata_rna.obs_names_make_unique()
    adata_msi.obs_names_make_unique()

    # Split into train/test without reordering
    rna_train = adata_rna[adata_rna.obs[split_name] == "train"].copy()
    rna_test = adata_rna[adata_rna.obs[split_name] == "test"].copy()

    # Save original names and rename for split tracking
    rna_train.obs["og_index"] = rna_train.obs_names
    rna_test.obs["og_index"] = rna_test.obs_names
    rna_train.obs_names = rna_train.obs["og_index"].astype(str) + "_11"
    rna_test.obs_names = rna_test.obs["og_index"].astype(str) + "_22"

    # Concatenate train + test for graph construction
    adata_temp = sc.concat([rna_train, rna_test])
    adata_temp.obs_names_make_unique()

    # Spatial graph feature extraction
    graph_features = []
    for slide in adata_temp.obs["slide"].unique():
        idx = np.where(adata_temp.obs["slide"] == slide)[0]
        adata_slide = adata_temp[idx].copy()

        sq.gr.spatial_neighbors(adata_slide,coord_type="grid",spatial_key="spatial",n_neighs=n_neighbors)
        svd = TruncatedSVD(n_components=n_components_graph, random_state=666)
        svd_graph = svd.fit_transform(adata_slide.obsp["spatial_connectivities"])
        
        df_slide = pd.DataFrame(svd_graph, index=adata_slide.obs_names)
        graph_features.append(df_slide)
    
    graph_df = pd.concat(graph_features)
    graph_df = graph_df.loc[adata_temp.obs_names]
    graph_feat_train = graph_df.loc[rna_train.obs_names].values
    graph_feat_test = graph_df.loc[rna_test.obs_names].values

    # RNA SVD
    svd = TruncatedSVD(n_components=n_components, random_state=666)
    svd_features_train = svd.fit_transform(rna_train.X.toarray())
    svd_features_test = svd.transform(rna_test.X.toarray())
    rna_train.obsm["svd_features"] = svd_features_train
    rna_test.obsm["svd_features"] = svd_features_test

    # Combine & scale features
    sc_svd = StandardScaler()
    sc_graph = StandardScaler()
    rna_train.obsm["svd_graph"] = np.concatenate([sc_svd.fit_transform(svd_features_train), sc_graph.fit_transform(graph_feat_train)], axis=1)
    rna_test.obsm["svd_graph"] = np.concatenate([sc_svd.fit_transform(svd_features_test),sc_graph.fit_transform(graph_feat_test)], axis=1)

    # Remove internal tracking column to avoid write conflicts and preserve index alignment
    rna_train.obs.drop(columns=["og_index"], inplace=True)
    rna_test.obs.drop(columns=["og_index"], inplace=True)
    #----------------------------------------------MSI----------------------------------------------#
    # Filter for highly variable metabolites
    if "highly_variable" not in adata_msi.var.columns:
        sc.pp.highly_variable_genes(adata_msi, flavor='seurat', n_top_genes=top_mets)
    
    hvg_msi = adata_msi[:, adata_msi.var["highly_variable"]].copy()
    hvg_msi_train = hvg_msi[hvg_msi.obs[split_name] == "train"].copy()
    hvg_msi_test = hvg_msi[hvg_msi.obs[split_name] == "test"].copy()

    # Save
    rna_train.write(output_rna_train)
    rna_test.write(output_rna_test)
    hvg_msi_train.write(output_msi_train)
    hvg_msi_test.write(output_msi_test)
