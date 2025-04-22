import scanpy as sc
import numpy as np
import squidpy as sq
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import StandardScaler
import pandas as pd


def process(
    adata_rna,
    adata_msi,
    output_rna_train,
    output_rna_test,
    output_msi_train,
    output_msi_test,
    split,
    params=None
):
    params = params or {}

    top_genes = params.get("top_genes", 2000)
    n_components = params.get("n_components", 50)
    n_neighbors = params.get("n_neighbors", 6)
    split_name = split

    # Ensure unique names early (preserves order)
    adata_rna.obs_names_make_unique()
    adata_msi.obs_names_make_unique()

    # --- HVG selection (RNA) ---
    if "highly_variable" not in adata_rna.var.columns:
        sc.pp.highly_variable_genes(
            adata_rna,
            flavor='seurat',
            n_top_genes=top_genes
        )
    hvg_rna = adata_rna[:, adata_rna.var["highly_variable"]].copy()

    # --- Split into train/test slices (order preserved) ---
    rna_train = hvg_rna[hvg_rna.obs[split_name] == "train"].copy()
    rna_test = hvg_rna[hvg_rna.obs[split_name] == "test"].copy()

    # --- Save original names for later and rename for concatenation ---
    rna_train.obs["og_index"] = rna_train.obs_names.astype(str)
    rna_test.obs["og_index"] = rna_test.obs_names.astype(str)
    rna_train.obs_names = rna_train.obs["og_index"] + "_11"
    rna_test.obs_names = rna_test.obs["og_index"] + "_22"

    # --- Concatenate for slide-wise graph building ---
    adata_temp = sc.concat([rna_train, rna_test])
    adata_temp.obs_names_make_unique()

    # --- Spatial graph feature extraction ---
    if (
        "spatial_connectivities" in rna_train.obsp
        and "spatial_connectivities" in rna_test.obsp
    ):
        # Use existing connectivities directly
        graph_feat_train = TruncatedSVD(
            n_components=n_components,
            random_state=666
        ).fit_transform(rna_train.obsp["spatial_connectivities"])
        graph_feat_test = TruncatedSVD(
            n_components=n_components,
            random_state=666
        ).fit_transform(rna_test.obsp["spatial_connectivities"])
    else:
        # Recompute per-slide spatial connectivities and SVD
        graph_features = []
        for slide in adata_temp.obs["slide"].unique():
            idx = np.where(adata_temp.obs["slide"] == slide)[0]
            adata_slide = adata_temp[idx].copy()
            sq.gr.spatial_neighbors(
                adata_slide,
                coord_type="grid",
                spatial_key="spatial",
                n_neighs=n_neighbors
            )
            svd_graph = TruncatedSVD(
                n_components=n_components,
                random_state=666
            ).fit_transform(
                adata_slide.obsp["spatial_connectivities"]
            )
            df_slide = pd.DataFrame(svd_graph, index=adata_slide.obs_names)
            graph_features.append(df_slide)
        graph_df = pd.concat(graph_features)
        graph_df = graph_df.loc[adata_temp.obs_names]
        graph_feat_train = graph_df.loc[rna_train.obs_names].values
        graph_feat_test = graph_df.loc[rna_test.obs_names].values

    # --- SVD on RNA.X for each split ---
    svd = TruncatedSVD(n_components=n_components, random_state=666)
    svd_features_train = svd.fit_transform(rna_train.X.toarray())
    svd_features_test = svd.fit_transform(rna_test.X.toarray())
    rna_train.obsm["svd_features"] = svd_features_train
    rna_test.obsm["svd_features"] = svd_features_test

    # --- Combine & scale SVD and graph features ---
    sc_svd = StandardScaler()
    sc_graph = StandardScaler()
    rna_train.obsm["svd_graph"] = np.concatenate([
        sc_svd.fit_transform(svd_features_train),
        sc_graph.fit_transform(graph_feat_train)
    ], axis=1)
    rna_test.obsm["svd_graph"] = np.concatenate([
        sc_svd.fit_transform(svd_features_test),
        sc_graph.fit_transform(graph_feat_test)
    ], axis=1)

    # --- Cleanup temporary columns to preserve original obs_names alignment ---
    rna_train.obs.drop(columns=["og_index"], inplace=True)
    rna_test.obs.drop(columns=["og_index"], inplace=True)

    # --- MSI HVG selection and split ---
    if "highly_variable" not in adata_msi.var.columns:
        sc.pp.highly_variable_genes(
            adata_msi,
            flavor='seurat',
            n_top_genes=top_genes
        )
    hvg_msi = adata_msi[:, adata_msi.var["highly_variable"]].copy()
    msi_train = hvg_msi[hvg_msi.obs[split_name] == "train"].copy()
    msi_test = hvg_msi[hvg_msi.obs[split_name] == "test"].copy()

    # --- Save outputs (order intact) ---
    rna_train.write(output_rna_train)
    rna_test.write(output_rna_test)
    msi_train.write(output_msi_train)
    msi_test.write(output_msi_test)

    return (rna_train, rna_test, msi_train, msi_test)
