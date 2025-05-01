import scanpy as sc

def process(adata_rna, adata_msi, output_rna_train, output_rna_test, output_msi_train, output_msi_test, split, params=None):
    

    split_name = split
    rna_train = adata_rna[adata_rna.obs[split_name] == "train"].copy()
    rna_test = adata_rna[adata_rna.obs[split_name] == "test"].copy()
    #----------------------------------------------MSI----------------------------------------------#
    msi_train = adata_msi[adata_msi.obs[split_name] == "train"].copy()
    msi_test = adata_msi[adata_msi.obs[split_name] == "test"].copy()

    #----------------------------------------------SAVE----------------------------------------------#
    rna_train.write(output_rna_train)
    rna_test.write(output_rna_test)
    msi_train.write(output_msi_train)
    msi_test.write(output_msi_test)
