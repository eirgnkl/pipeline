import scanpy as sc
import numpy as np
import squidpy as sq
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import StandardScaler


def process(adata_rna, adata_msi, output_rna_train, output_rna_test, output_msi_train, output_msi_test, params=None):
    params = params or {}

    top_genes = params.get("top_genes", 5000)
    n_components = params.get("n_components", 16)
    n_neighbors = params.get("n_neighbors", 6)
    split_name = params.get("split_name", "split")

    #----------------------------------------------sc-seqRNA----------------------------------------------#
    #-----HVG-----#
    if "highly_variable" not in adata_rna.var.columns:
        sc.pp.highly_variable_genes(adata_rna, flavor='seurat', n_top_genes=top_genes)
    hvg_rna = adata_rna[:, adata_rna.var["highly_variable"]]


    #split train and test, used for svd and rest
    hvg_rna_train = hvg_rna[hvg_rna.obs[split_name] == "train"]
    hvg_rna_test = hvg_rna[hvg_rna.obs[split_name] == "test"]

    #-----SVD-----#
    svd_reducer = TruncatedSVD(n_components=n_components) 

    svd_features_train = svd_reducer.fit_transform(hvg_rna_train.X.toarray())
    hvg_rna_train.obsm["svd_features"] = svd_features_train

    svd_features_test = svd_reducer.fit_transform(hvg_rna_test.X.toarray())
    hvg_rna_test.obsm["svd_features"] = svd_features_test

    #-----GRAPH-----#
    ##Check if og data has spatial connectivities obsp else do the process here:


    graph_feat_train = svd_reducer.fit_transform(hvg_rna_train[hvg_rna_train.obs_names].obsp["spatial_connectivities"])
    graph_feat_test = svd_reducer.fit_transform(hvg_rna_test[hvg_rna_test.obs_names].obsp["spatial_connectivities"])


    sc_svd = StandardScaler()
    sc_gr = StandardScaler()

    #Concatentate the features 
    rna_hsg_train = np.concatenate([sc_svd.fit_transform(svd_features_train), \
                                            sc_gr.fit_transform(graph_feat_train)],
                                         axis=1)
    hvg_rna_train.obsm["svd_graph"] = rna_hsg_train

    rna_hsg_test = np.concatenate([sc_svd.fit_transform(svd_features_test), \
                                            sc_gr.fit_transform(graph_feat_test)],
                                         axis=1)
    hvg_rna_test.obsm["svd_graph"] = rna_hsg_test

    #----------------------------------------------MSI----------------------------------------------#
    #-----HVG-----#
    #MSI processed only for highly variable metabolites, kept hvg_ for uniformality in vars
    if "highly_variable" not in adata_msi.var.columns:
        sc.pp.highly_variable_genes(adata_msi, flavor='seurat', n_top_genes=top_genes)
    hvg_msi = adata_msi[:, adata_msi.var["highly_variable"]]

    hvg_msi_train = hvg_msi[hvg_msi.obs[split_name] == "train"]
    hvg_msi_test = hvg_msi[hvg_msi.obs[split_name] == "test"]

    #----------------------------------------------SAVE----------------------------------------------#
    hvg_rna_train.write(output_rna_train)
    hvg_rna_test.write(output_rna_test)
    hvg_msi_train.write(output_msi_train)
    hvg_msi_test.write(output_msi_test)
