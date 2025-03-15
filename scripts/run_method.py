# import time
# start_time = time.time()

import scanpy as sc # type: ignore
import pandas as pd # type: ignore
import ast
import os

# Import all method scripts
from ridge import run_ridge_reg
from linear import run_linreg
from lasso import run_lasso
from mxgboost import run_xgboost
from elastic_net import run_elastic_net
from cvae import run_cvae
from gnn import run_gnn

# Dictionary mapping method names to their functions - connect config->smk->run_methods->function.py 
METHOD_MAP = {
    'ridge': dict(function=run_ridge_reg, mode='paired'),
    'lasso': dict(function=run_lasso, mode='paired'),
    'linear': dict(function=run_linreg, mode='paired'),
    'xgboost': dict(function=run_xgboost, mode='paired'),
    'elastic_net': dict(function=run_elastic_net, mode='paired'),
    'cvae': dict(function=run_cvae, mode='paired'),
    'gnn': dict(function=run_gnn, mode='paired')

}

# Load parameters from Snakemake
params = snakemake.params.thisparam 
input_rna_train = snakemake.input.rna_ds_train
input_rna_test = snakemake.input.rna_ds_test
input_msi_train = snakemake.input.msi_ds_train
input_msi_test = snakemake.input.msi_ds_test

if all(os.path.exists(f) for f in snakemake.output):
    print(f"Skipping {snakemake.wildcards.task} - {snakemake.wildcards.method}, results already exist")
    exit(0)

# Task parameters
method = params['method']
task = params['task']
hash_id = params['hash']
featsel = params['featsel']

method_params = ast.literal_eval(params['params'])
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

    metrics_df, predictions_df = method_function(
        adata_rna_train=adata_rna_train,
        adata_rna_test=adata_rna_test,
        adata_msi_train=adata_msi_train,
        adata_msi_test=adata_msi_test, 
        params=method_params,
        featsel=featsel
    )


# Add metadata to the results and save to output file
metrics_df['task'] = task
metrics_df['method_name'] = method
metrics_df['featsel'] = featsel
metrics_df['method_params'] = str(method_params)
metrics_df['hash'] = hash_id
metrics_df.to_csv(snakemake.output.metrics, sep='\t', index=False)
predictions_df.to_csv(snakemake.output.predictions, sep='\t', index=False)