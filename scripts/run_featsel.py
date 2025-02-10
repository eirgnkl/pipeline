import importlib.util
import pandas as pd
import os
import scanpy as sc

# Load Snakemake variables
task = snakemake.wildcards.task
featsel = snakemake.wildcards.featsel
output_rna = snakemake.output.rna_ds
output_msi = snakemake.output.msi_ds
tasks_df = pd.read_csv(snakemake.input.tasks_df, sep="\t")

# Get input datasets
task_row = tasks_df[(tasks_df["task"] == task) & (tasks_df["featsel"] == featsel)].iloc[0]
input_rna = task_row["input_rna"]
input_metabolomics = task_row["input_metabolomics"]

# Dynamically load the feature selection script
feat_sel_script = snakemake.params.featsel_script
spec = importlib.util.spec_from_file_location("feature_selection", feat_sel_script)
feature_selection_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(feature_selection_module)

# Ensure output directories exist
os.makedirs(os.path.dirname(output_rna), exist_ok=True)
os.makedirs(os.path.dirname(output_msi), exist_ok=True)


# Load tuning parameters from the tuning file (assumed TSV)
tuning_file = snakemake.input.tuning
tuning_df = pd.read_csv(tuning_file, sep="|")

# Filter parameters for the current feature selection method and convert to dict
tuning_dict = tuning_df[tuning_df["method"] == featsel].set_index("parameter")["value"].to_dict()

# Run feature selection for RNA and MSI
adata_rna = sc.read_h5ad(input_rna)
adata_msi = sc.read_h5ad(input_metabolomics)

feature_selection_module.process(adata_rna, adata_msi, output_rna, output_msi, params=tuning_dict)