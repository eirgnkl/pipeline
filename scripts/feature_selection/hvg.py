##Important: if your dataset is preprocessed and has already marked highly 
##variable, the name in stored in var MUST BE "highly variable", else don't 
##forget to change it in these scripts

import scanpy as sc

def process(adata_rna, adata_msi, output_rna_train, output_rna_test, output_msi_train, output_msi_test, split, params=None):
    
    #Use an empty dict in case no params avail
    params = params or {}
    top_genes = params.get("top_genes", 2000)
    split_name = split


    #-----HVG-----#
    if "highly_variable" not in adata_rna.var.columns:
        sc.pp.highly_variable_genes(adata_rna, flavor='seurat', n_top_genes=top_genes)
    hvg_rna = adata_rna[:, adata_rna.var["highly_variable"]].copy()
    
    #split train and test, used for svd and rest
    hvg_rna_train = hvg_rna[hvg_rna.obs[split_name] == "train"].copy()
    hvg_rna_test = hvg_rna[hvg_rna.obs[split_name] == "test"].copy()

    #----------------------------------------------MSI----------------------------------------------#
    #MSI processed only for highly variable metabolites, kept hvg_ for uniformality in vars
    if "highly_variable" not in adata_msi.var.columns:
        sc.pp.highly_variable_genes(adata_msi, flavor='seurat', n_top_genes=top_genes)
    hvg_msi = adata_msi[:, adata_msi.var["highly_variable"]].copy()

    hvg_msi_train = hvg_msi[hvg_msi.obs[split_name] == "train"].copy()
    hvg_msi_test = hvg_msi[hvg_msi.obs[split_name] == "test"].copy()

    #----------------------------------------------SAVE----------------------------------------------#
    hvg_rna_train.write(output_rna_train)
    hvg_rna_test.write(output_rna_test)
    hvg_msi_train.write(output_msi_train)
    hvg_msi_test.write(output_msi_test)

