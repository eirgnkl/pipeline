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


# # --- Added: Skip if outputs already exist ---
# outputs = [output_rna_train, output_rna_test, output_msi_train, output_msi_test]
# if all(os.path.exists(f) for f in outputs):
#     print("All output files already exist. Skipping feature selection.", flush=True)
#     exit(0)

tasks_df = pd.read_csv(snakemake.input.tasks_df, sep="\t")

#Extract correct datasets for input by figuring out the line where task and featsel are the correct ones
task_row = tasks_df[(tasks_df["task"] == task) & (tasks_df["featsel"] == featsel)].iloc[0]
input_rna = task_row["input_rna"]
input_metabolomics = task_row["input_metabolomics"]
split = task_row["split"]

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

#Read the tuning parameters from tuning.tsv
tuning_file =snakemake.input.tuning
tuning_df = pd.read_csv(tuning_file, sep="|")

#Extract params
# Extract all tuning parameters for the given method and build a parameter dictionary.
if featsel in tuning_df["method"].values:
    featsel_params_df = tuning_df[tuning_df["method"] == featsel]
    # Create a dictionary with parameters as keys and values from the tuning file.
    tuning_params = dict(zip(featsel_params_df["parameter"], featsel_params_df["value"]))
else:
    print(f"[Warning] No tuning parameters found for featsel={featsel}. Proceeding with empty dict.")
    tuning_params = {}

# Pass the parameters into the process function
feature_selection_module.process(
    adata_rna, adata_msi,
    output_rna_train, output_rna_test,
    output_msi_train, output_msi_test,
    split,
    params=tuning_params
)

