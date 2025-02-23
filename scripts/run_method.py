# import time
# start_time = time.time()

import scanpy as sc # type: ignore
import pandas as pd # type: ignore
import ast


# Import all method scripts
from ridge import run_ridge_reg
from linear import run_linreg
from lasso import run_lasso
from mxgboost import run_xgboost
from elastic_net import run_elastic_net
from cvae import run_cvae

# Dictionary mapping method names to their functions - connect config->smk->run_methods->function.py 
METHOD_MAP = {
    'ridge': dict(function=run_ridge_reg, mode='paired'),
    'lasso': dict(function=run_lasso, mode='paired'),
    'linear': dict(function=run_linreg, mode='paired'),
    'xgboost': dict(function=run_xgboost, mode='paired'),
    'elastic_net': dict(function=run_elastic_net, mode='paired'),
    'cvae': dict(function=run_cvae, mode='paired')
}

# Load parameters from Snakemake
params = snakemake.params.thisparam 
input_rna_train = snakemake.input.rna_ds_train
input_rna_test = snakemake.input.rna_ds_test
input_msi_train = snakemake.input.msi_ds_train
input_msi_test = snakemake.input.msi_ds_test


 # Task parameters
method = params['method']
task = params['task']
hash_id = params['hash']
featsel = params['featsel']
# input_rna = input['input_rna']
# input_metabolomics = params['input_metabolomics']
# processed_rna = f"dataset/processed/{task}/{featsel}/rna_dataset.h5ad"
# processed_msi = f"dataset/processed/{task}/{featsel}/msi_dataset.h5ad"
# output_method_params = params['params']

method_params = ast.literal_eval(params['params'])
output_file = snakemake.output.tsv
# Load the appropriate method function
method_mode = METHOD_MAP[method]['mode']
method_function = METHOD_MAP[method]['function']

# Load data based on method mode
if method_mode == 'paired':
    
    #pass extracted input path to create each ds
    adata_rna_train = sc.read_h5ad(input_rna_train)
    adata_rna_test = sc.read_h5ad(input_rna_test)
    adata_msi_train = sc.read_h5ad(input_msi_train)
    adata_msi_test = sc.read_h5ad(input_msi_test)

    result_df = method_function(
        adata_rna_train=adata_rna_train,
        adata_rna_test=adata_rna_test,
        adata_msi_train=adata_msi_train,
        adata_msi_test=adata_msi_test, 
        params=method_params,
        featsel=featsel
        )

# Add metadata to the results and save to output file
result_df['task'] = task
result_df['method_name'] = method
result_df['featsel'] = featsel
result_df['method_params'] = str(method_params)
result_df['hash'] = hash_id
result_df.to_csv(output_file, sep='\t', index=False)