import scanpy as sc
import numpy as np
import squidpy as sq
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import StandardScaler


def process(adata_rna, adata_msi, output_rna_train, output_rna_test, output_msi_train, output_msi_test, split, params=None):

    #Extract input params
    params = params or {}
    n_components = params.get("n_components", 16)
    n_neighbors = params.get("n_neighbors", 6)
    split_name = split
    adata_rna.obs_names_make_unique()
    adata_msi.obs_names_make_unique()

    #Split into train test
    rna_train = adata_rna[adata_rna.obs[split_name] == "train"].copy()
    rna_test = adata_rna[adata_rna.obs[split_name] == "test"].copy()

    #----------------------------------------------sc-seqRNA----------------------------------------------#
    #-----SVD-----#
    svd_reducer = TruncatedSVD(n_components=n_components)

    svd_features_train = svd_reducer.fit_transform(rna_train.X.toarray())
    rna_train.obsm["svd_features"] = svd_features_train

    svd_features_test = svd_reducer.fit_transform(rna_test.X.toarray())
    rna_test.obsm["svd_features"] = svd_features_test

    #-----GRAPH-----#
    # #Check if og data has spatial connectivities or else create the connectivity matrix 
    if "spatial_connectivities" not in rna_train.obsp:
        rna_train.obs_names_make_unique()
        rna_test.obs_names_make_unique()

        rna_train.obs_names = rna_train.obs.og_index.tolist().copy()
        rna_train.obs_names_make_unique()
        rna_train.obs_names = rna_train.obs_names + "_11"

        rna_test.obs_names = rna_test.obs.og_index.tolist().copy()
        rna_test.obs_names_make_unique()
        rna_test.obs_names = rna_test.obs_names + "_22"
        adata_temp = sc.concat([rna_train, rna_test])
        sq.gr.spatial_neighbors(adata_temp, coord_type="grid", spatial_key="spatial", n_neighs=n_neighbors)
        svd_reducer = TruncatedSVD(n_components=n_components)

        graph_feat_train = svd_reducer.fit_transform(adata_temp[rna_train.obs_names].obsp["spatial_connectivities"])
        graph_feat_test = svd_reducer.fit_transform(adata_temp[rna_test.obs_names].obsp["spatial_connectivities"])
    else:
         svd_reducer = TruncatedSVD(n_components=n_components)
         graph_feat_train = svd_reducer.fit_transform(rna_train[rna_train.obs_names].obsp["spatial_connectivities"])
         graph_feat_test = svd_reducer.fit_transform(rna_test[rna_test.obs_names].obsp["spatial_connectivities"])

    ##Concatenate the standardized features as obtained by svd applied on adata.X and on the s
    sc_svd = StandardScaler()
    sc_gr = StandardScaler()

    rna_sg_train = np.concatenate([sc_svd.fit_transform(svd_features_train), \
                                            sc_gr.fit_transform(graph_feat_train)],
                                         axis=1)
    rna_train.obsm["svd_graph"] = rna_sg_train

    rna_sg_test = np.concatenate([sc_svd.fit_transform(svd_features_test), \
                                            sc_gr.fit_transform(graph_feat_test)],
                                         axis=1)
    rna_test.obsm["svd_graph"] = rna_sg_test

    #----------------------------------------------MSI----------------------------------------------#
    #-----no process-----#
    msi_train = adata_msi[adata_msi.obs[split_name] == "train"].copy()
    msi_test = adata_msi[adata_msi.obs[split_name] == "test"].copy()

    #----------------------------------------------SAVE----------------------------------------------#
    rna_train.write(output_rna_train)
    rna_test.write(output_rna_test)
    msi_train.write(output_msi_train)
    msi_test.write(output_msi_test)
