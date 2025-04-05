import pandas as pd
from snakemake.utils import Paramspace
from scripts.utils import create_tasks_df
from pprint import pprint
import numpy as np
import os
import time


# Instructions for user:
# 1. Make sure to select your parameters for the models in folder params by naming your file according to model_params.tsv
# 2. Similarly for feature selection, make sure to edit the scripts/feature_selection/tuning.tsv to accomodate your preferences
# Be careful, so far there is no dynamic selection for feature selection. You can specify your parameters, but not have multiple
# versions of the same preprocessing method

def safe_read_csv(path, max_wait=30, interval=1):
    waited = 0
    while waited < max_wait:
        if os.path.exists(path) and os.path.getsize(path) > 0:
            try:
                return pd.read_csv(path, sep='\t')
            except pd.errors.EmptyDataError:
                pass  # Try again
        time.sleep(interval)
        waited += interval
    raise RuntimeError(f"File '{path}' exists but is still empty or unreadable after {max_wait} seconds.")

os.makedirs("data", exist_ok=True)
tasks_df = create_tasks_df('config.yaml', save='data/tasks.tsv')
tasks_df = safe_read_csv('data/tasks.tsv')

# Extract unique task details
hashes = tasks_df['hash'].unique()
methods = tasks_df['method'].unique()
tasks = tasks_df['task'].unique()


#adding .str.strip().tolist(), because otherwise versions of snakemake <8 mess this up
rule all:
    input:
        #RNA datasets (train/test)
        expand("dataset/processed/{task}/{featsel}/rna_dataset_train.h5ad",
               zip,
               task=tasks_df['task'].str.strip().tolist(),
               featsel=tasks_df['featsel'].str.strip().tolist()),
        expand("dataset/processed/{task}/{featsel}/rna_dataset_test.h5ad",
               zip,
               task=tasks_df['task'].str.strip().tolist(),
               featsel=tasks_df['featsel'].str.strip().tolist()),
        #MSI datasets (train/test)
        expand("dataset/processed/{task}/{featsel}/msi_dataset_train.h5ad",
               zip,
               task=tasks_df['task'].str.strip().tolist(),
               featsel=tasks_df['featsel'].str.strip().tolist()),
        expand("dataset/processed/{task}/{featsel}/msi_dataset_test.h5ad",
               zip,
               task=tasks_df['task'].str.strip().tolist(),
               featsel=tasks_df['featsel'].str.strip().tolist()),
        #Accuracy reports
        expand("data/reports/{task}/{method}/{featsel}/{hash}/accuracy.tsv",
               zip,
               task=tasks_df['task'].str.strip().tolist(),
               method=tasks_df['method'].str.strip().tolist(),
               featsel=tasks_df['featsel'].str.strip().tolist(),
               hash=tasks_df['hash'].str.strip().tolist()),
        #Merge for each task
        expand("data/reports/{task}/merged_results.tsv",
               task=[t.strip() for t in tasks_df['task'].unique()]),
        #Per model best RMSE and R2
        expand("data/reports/{task}/best_results_per_model_rmse.tsv",
               task=[t.strip() for t in tasks_df['task'].unique()]),
        expand("data/reports/{task}/best_results_per_model_r2.tsv",
               task=[t.strip() for t in tasks_df['task'].unique()]),
        #Top 10 best
        expand("data/reports/{task}/best_results_overall_rmse.tsv",
               task=[t.strip() for t in tasks_df['task'].unique()]),
        expand("data/reports/{task}/best_results_overall_r2.tsv",
               task=[t.strip() for t in tasks_df['task'].unique()]),
        #Visualisation of metrics results
        expand("data/reports/{task}/metrics_visualisation_{task}.png",
               task=[t.strip() for t in tasks_df['task'].unique()])



rule feat_sel:
    input:
        tasks_df="data/tasks.tsv",
        tuning="scripts/feature_selection/tuning.tsv"
    output:
        rna_ds_train="dataset/processed/{task}/{featsel}/rna_dataset_train.h5ad",
        rna_ds_test="dataset/processed/{task}/{featsel}/rna_dataset_test.h5ad",
        msi_ds_train="dataset/processed/{task}/{featsel}/msi_dataset_train.h5ad",
        msi_ds_test="dataset/processed/{task}/{featsel}/msi_dataset_test.h5ad"
    params:
        featsel_script="scripts/feature_selection/{featsel}.py"
    script:
        "scripts/run_featsel.py"

rule run_method:
    input:
        rna_ds_train="dataset/processed/{task}/{featsel}/rna_dataset_train.h5ad",
        rna_ds_test="dataset/processed/{task}/{featsel}/rna_dataset_test.h5ad",
        msi_ds_train="dataset/processed/{task}/{featsel}/msi_dataset_train.h5ad",
        msi_ds_test="dataset/processed/{task}/{featsel}/msi_dataset_test.h5ad"
    output:
        metrics="data/reports/{task}/{method}/{featsel}/{hash}/accuracy.tsv",
        predictions="data/reports/{task}/{method}/{featsel}/{hash}/predictions.tsv"
    params:
        thisparam=lambda wildcards: tasks_df.loc[tasks_df['hash'] == wildcards.hash, :].iloc[0, :].to_dict()
    script:
        "scripts/run_method.py"

rule merge:
    input:
        lambda wildcards: expand(
            "data/reports/{task}/{method}/{featsel}/{hash}/accuracy.tsv",
            zip,
            task=[wildcards.task] * len(tasks_df[tasks_df['task'] == wildcards.task]),
            method=tasks_df[tasks_df['task'] == wildcards.task]['method'].tolist(),
            featsel=tasks_df[tasks_df['task'] == wildcards.task]['featsel'].tolist(),
            hash=tasks_df[tasks_df['task'] == wildcards.task]['hash'].tolist()
        )
    output:
        tsv="data/reports/{task}/merged_results.tsv"
    run:
        dfs = [pd.read_csv(file, sep='\t') for file in input if os.path.exists(file)]
        merged_df = pd.concat(dfs)
        merged_df.to_csv(output.tsv, sep='\t', index=False)

rule find_best:
    input:
        tsv="data/reports/{task}/merged_results.tsv"
    output:
        per_model_rmse="data/reports/{task}/best_results_per_model_rmse.tsv",
        per_model_r2="data/reports/{task}/best_results_per_model_r2.tsv",
        overall_rmse="data/reports/{task}/best_results_overall_rmse.tsv",
        overall_r2="data/reports/{task}/best_results_overall_r2.tsv"
    run:
        import pandas as pd
        df = pd.read_csv(input.tsv, sep='\t')
        
        # -- Per-model Best Selection --
        per_model_rmse_rows = []
        per_model_r2_rows = []
        grouped = df.groupby(['method_name', 'task'])
        
        for (method_name, task), group in grouped:
            if 'rmse' in group.columns:
                # For RMSE, lower values are better
                best_row_rmse = group.loc[group['rmse'].idxmin()]
                per_model_rmse_rows.append(best_row_rmse)
            if 'r2' in group.columns:
                # For R², higher values are better
                best_row_r2 = group.loc[group['r2'].idxmax()]
                per_model_r2_rows.append(best_row_r2)
        
        best_per_model_rmse_df = pd.DataFrame(per_model_rmse_rows).drop_duplicates()
        best_per_model_r2_df = pd.DataFrame(per_model_r2_rows).drop_duplicates()
        best_per_model_rmse_df = best_per_model_rmse_df.sort_values(by='rmse', ascending=True)
        best_per_model_r2_df = best_per_model_r2_df.sort_values(by='r2', ascending=False)

        # -- Overall Top 10 Selection --
        best_overall_rmse_df = df.sort_values(by='rmse', ascending=True).head(10)
        best_overall_r2_df = df.sort_values(by='r2', ascending=False).head(10)
        
        best_per_model_rmse_df.to_csv(output.per_model_rmse, sep='\t', index=False)
        best_per_model_r2_df.to_csv(output.per_model_r2, sep='\t', index=False)
        best_overall_rmse_df.to_csv(output.overall_rmse, sep='\t', index=False)
        best_overall_r2_df.to_csv(output.overall_r2, sep='\t', index=False)

rule visualize:
    input:
        merged_results="data/reports/{task}/merged_results.tsv"
    output:
        visualization="data/reports/{task}/metrics_visualisation_{task}.png"
    params:
        top_model=5  # Set the number of top models to display
    script:
        "scripts/metrics_visualization.py"
