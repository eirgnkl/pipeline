import pandas as pd
from snakemake.utils import Paramspace
from scripts.utils import create_tasks_df
from pprint import pprint
import numpy as np
import os

# Instructions for user:
# 1. Make sure to select your parameters for the models in folder params by naming your file according to model_params.tsv
# 2. Similarly for feature selection, make sure to edit the scripts/feature_selection/tuning.tsv to accomodate your preferences
# Be careful, so far there is no dynamic selection for feature selection. You can specify your parameters, but not have multiple
# versions of the same preprocessing method



# Generate tasks DataFrame and load configuration
os.makedirs("data", exist_ok=True)
tasks_df = create_tasks_df('config.yaml', save='data/tasks.tsv')
tasks_df = pd.read_csv('data/tasks.tsv', sep='\t')

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
        # Merged and best results per unique task
        expand("data/reports/{task}/merged_results.tsv",
               task=[t.strip() for t in tasks_df['task'].unique()]),
        expand("data/reports/{task}/best_results.tsv",
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
        'scripts/run_method.py'


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
        tsv="data/reports/{task}/best_results.tsv"
    run:
        import pandas as pd
        df = pd.read_csv(input.tsv, sep='\t')
        best_rows = []
        grouped = df.groupby(['method_name', 'task'])

        # Iterate through groups, find best rows
        for (method_name, task), group in grouped:
            for metric in ['rmse', 'r2', 'pearson', 'spearman']:
                if metric in group.columns:
                    best_row = group.loc[group[metric].idxmax() if metric != 'rmse' else group[metric].idxmin()]
                    best_rows.append(best_row)

        best_results_df = pd.DataFrame(best_rows).drop_duplicates()
        best_results_df.to_csv(output.tsv, sep='\t', index=False)