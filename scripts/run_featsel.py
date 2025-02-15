import importlib.util
import pandas as pd
import os
import scanpy as sc

#Load Snakemake variables
#task and featsel are wildcards [just like hash and model]
task = snakemake.wildcards.task
featsel = snakemake.wildcards.featsel
output_rna_train = snakemake.output.rna_ds_train
output_rna_test = snakemake.output.rna_ds_test
output_msi_train = snakemake.output.msi_ds_train
output_msi_test = snakemake.output.msi_ds_test

tasks_df = pd.read_csv(snakemake.input.tasks_df, sep="\t")

#Extract correct datasets for input by figuring out the line where task and featsel are the correct ones
task_row = tasks_df[(tasks_df["task"] == task) & (tasks_df["featsel"] == featsel)].iloc[0]
input_rna = task_row["input_rna"]
input_metabolomics = task_row["input_metabolomics"]

# Dynamically load the feature selection script
feat_sel_script = snakemake.params.featsel_script
spec = importlib.util.spec_from_file_location("feature_selection", feat_sel_script)
feature_selection_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(feature_selection_module)

# Ensure output directories exist
os.makedirs(os.path.dirname(output_rna_train), exist_ok=True)
os.makedirs(os.path.dirname(output_rna_test), exist_ok=True)
os.makedirs(os.path.dirname(output_msi_train), exist_ok=True)
os.makedirs(os.path.dirname(output_msi_test), exist_ok=True)


#Set correct datasets
adata_rna = sc.read_h5ad(input_rna)
adata_msi = sc.read_h5ad(input_metabolomics)

feature_selection_module.process(adata_rna, adata_msi, output_rna_train, output_rna_test, output_msi_train, output_msi_test)