##Important: if your dataset is preprocessed and has already marked highly 
##variable, the name in stored in var MUST BE "highly variable", else don't 
##forget to change it here

import scanpy as sc

def process(adata_rna, adata_msi, output_rna, output_msi, params=None):
    # Use an empty dict if no params avail
    params = params or {}

    top_genes = params.get("top_genes", 5000)

    if "highly_variable" not in adata_rna.var.columns:
            sc.pp.highly_variable_genes(adata_rna, flavor='seurat', n_top_genes=top_genes)

    hvg_rna = adata_rna.var[adata_rna.var["highly_variable"]]
    selected_genes_rna = hvg_rna.index
    highly_var_rna = adata_rna[:, selected_genes_rna].copy()

    if "highly_variable" not in adata_msi.var.columns:
            sc.pp.highly_variable_genes(adata_msi, flavor='seurat', n_top_genes=top_genes)             
    hvm_msi = adata_msi.var[adata_msi.var["highly_variable"]]
    selected_genes_msi = hvm_msi.index
    highly_var_msi = adata_msi[:, selected_genes_msi].copy()
    
    #Save   
    highly_var_rna.write(output_rna)
    highly_var_msi.write(output_msi)
