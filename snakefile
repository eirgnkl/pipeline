import pandas as pd
from snakemake.utils import Paramspace
from scripts.utils import create_tasks_df
from pprint import pprint
import numpy as np

## Current script can process only most 

# Instructions for user:
# 1. Make sure to select your parameters for the models in folder params by naming your file according to model_params.tsv
# 2. Similarly for feature selection, make sure to edit the scripts/feature_selection/tuning.tsv to accomodate your preferences
# Be careful, so far there is no dynamic selection for feature selection. You can specify your parameters, but not have multiple
# versions of the same preprocessing method

# Generate tasks DataFrame and load configuration
tasks_df = create_tasks_df('config.yaml', save='data/tasks.tsv')
tasks_df = pd.read_csv('data/tasks.tsv', sep='\t')

# Extract unique task details
hashes = tasks_df['hash'].unique()
methods = tasks_df['method'].unique()
tasks = tasks_df['task'].unique()

rule all:
    input:
        # Datasets
        [
            f"dataset/processed/{row['task']}/{row['featsel']}/rna_dataset.h5ad"
            for _, row in tasks_df.iterrows()
        ],
        [
            f"dataset/processed/{row['task']}/{row['featsel']}/msi_dataset.h5ad"
            for _, row in tasks_df.iterrows()
        ],
        # Reports
        [
            f"data/reports/{row['task']}/{row['featsel']}/{row['method']}/{row['hash']}/accuracy.tsv"
            for _, row in tasks_df.iterrows()
        ],
        # Ensure the merged and best results are generated
        'data/reports/merged_results.tsv',
        'data/reports/best_results.tsv'
        

rule feat_sel:
    input:
        tasks_df="data/tasks.tsv",
        tuning="scripts/feature_selection/tuning.tsv"
    output:
        rna_ds="dataset/processed/{task}/{featsel}/rna_dataset.h5ad",
        msi_ds="dataset/processed/{task}/{featsel}/msi_dataset.h5ad"
    params:
        featsel_script="scripts/feature_selection/{featsel}.py"
    script:
        "scripts/run_featsel.py"


rule run_method:
    output:
        tsv='data/reports/{task}/{featsel}/{method}/{hash}/accuracy.tsv'
    params:
        thisparam=lambda wildcards: tasks_df.loc[tasks_df['hash'] == wildcards.hash, :].iloc[0, :].to_dict()
    script:
        'scripts/run_method.py'


rule merge:
    input:
        expand(
            'data/reports/{task}/{featsel}/{method}/{hash}/accuracy.tsv',
            zip,
            task=tasks_df['task'].values,
            featsel=tasks_df['featsel'].values,
            method=tasks_df['method'].values,
            hash=tasks_df['hash'].values
        )
    output:
        tsv='data/reports/merged_results.tsv'
    run:
        dfs = [pd.read_csv(file, sep='\t') for file in input if os.path.exists(file)]
        merged_df = pd.concat(dfs)
        merged_df.to_csv(output.tsv, sep='\t', index=False)

rule find_best:
    input:
        tsv='data/reports/merged_results.tsv'
    output:
        tsv='data/reports/best_results.tsv'
    run:
        import pandas as pd
        df = pd.read_csv(input.tsv, sep='\t')
        best_rows = []
        grouped = df.groupby(['method_name', 'featsel', 'task'])

        # Iterate through groups, find best rows
        for (method_name, featsel, task), group in grouped:
            for metric in ['mse', 'r2', 'pearson', 'spearman']:
                if metric in group.columns:
                    best_row = group.loc[group[metric].idxmax() if metric != 'mse' else group[metric].idxmin()]
                    best_rows.append(best_row)

        best_results_df = pd.DataFrame(best_rows).drop_duplicates()
        best_results_df.to_csv(output.tsv, sep='\t', index=False)
