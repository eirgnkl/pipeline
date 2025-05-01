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
    msi_train = adata_msi[adata_msi.obs[split_name] == "train"].copy()
    msi_test = adata_msi[adata_msi.obs[split_name] == "test"].copy()

    #----------------------------------------------SAVE----------------------------------------------#
    hvg_rna_train.write(output_rna_train)
    hvg_rna_test.write(output_rna_test)
    msi_train.write(output_msi_train)
    msi_test.write(output_msi_test)
