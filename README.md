
## Pipeline for Metabolite Prediction from Gene Expression (with feature preprocessing)

This repository contains a Snakemake pipeline for predicting **metabolic distribution from gene expression data** using several **machine learning models** and **feature selection techniques**.


## Pipeline Structure

The pipeline is structured as follows:

1. **Feature Preprocessing**: Selects relevant features using different feature selection methods.
2. **Model Training**: Runs Ridge, Lasso, Linear Regression, XGBoost and a CVAE.
3. **Evaluation**: Assesses performance using 4 different metrics (RMSE, Pearson and Spearmann Correlation and $R^2$).

## Feature Preprocessing

The dataset undergoes multiple feature selection techniques to reduce dimensionality and improve model performance. The available methods are:

- **hvg**: High-variance genes.
- **hvg+svd**: High-variance genes + singular value decomposition.
- **hvg+svd+graph**: Adds graph-based selection.
- **svd**: Singular value decomposition.
- **svd+graph**: Graph-based selection on SVD processed data.

You can tune the parameters for the feature selection scripts (e.g. number of top genes, number of components for svd, neighbours to take into account for graph etc.) in `scripts/feature_selection/tuning.tsv .`

##### *Note:*

The feature selection process is not dynamic. This means that if you run multiple tasks (which probably means different datasets), your feature selection process will be the same for all the tasks. If you wish to have different tuning params for the feature selection, I recommend running the pipleine for the different tasks seperately (so leave only task of interest uncommented in the config file) and setting the `tuning.tsv` separately for each task.

#### !Warning!:

If you change the feature selection scripts for any reason, make sure to do **one of the two**:

* Comment out lines that skip creating the dataset if they already find existing file in the path (lines 17-20) in the `scripts/run_featsel.py`
* Delete manually the already existing datasets of your task. These are found in the directory `dataset/processed/{task}.`

## How to set Tasks, Feature Selection and Models:

Visit `config.yaml` and follow these steps:

1. Set name of `'task'` in `config.yaml` under the `TASKS` key.
2. Set correct paths to your RNA and MSI datasets in `input_rna` and `input_metabolomics.`
3. Define the name of the `split `that you want to use for the task (this is used later in both feature selection and training thus it's set up in the `config.yaml`)
4. Specify the `methods` (models) that you want to use to make predictions. For the parameter tuning keep reading.
5. Select the different ways of preprocessing that you want your models to run with, under the `featsel` key.

```
TASKS:
  'vitatrack':
    input_rna: /path_to_rna/rna_file.h5ad
    input_metabolomics: /path_to_msi/msi_file.h5ad
    split: split
    methods:
      ridge:
        params: params/ridge_params.tsv
        featsel:
          - hvg
          - hvg_svd

```

## Models Implemented

The pipeline supports the following regression models:

- **Ridge Regression**: Handles multicollinearity by adding an L2 penalty.
- **Lasso Regression**: Adds an L1 penalty for feature selection.
- **Linear Regression**: Standard least squares regression.
- **Elastic Net**: Apply both types of penalties, L1 and L2
- **XGBoost**: Gradient boosting for non-linear patterns.

In case you want to add new models, be aware that the models is called through the `run_methods.py`, so make sure the structure of it is similar to the already existing scripts and define a `{new_methods}_param.tsv`, in the folder `params`

### Hyperparameters to Tune

Each model has parameters that users can configure in `params/{method}_params.tsv`.

## Running the Pipeline

To execute the pipeline, use:

```bash
snakemake --cores <num_cores> --profile profile_gpu
```

For dry-run mode:

```bash
snakemake --dry-run
```

Be mindful to set profile_gpu/config.yaml to cluster needs


## Visualization

### Model Performance Visualization

After the pipeline completes, a **visualization step** generates comparative plots for each task. These plots provide a clear view of model performance across different **feature selection techniques**.

### Visualization Includes:

- 📊 **Bar charts** showing model performance for each metric (**RMSE, Pearson, Spearman, R²**).
- 🎯 **Feature selection methods displayed inside bars** instead of model parameters.
- ⭐ **Best-performing models highlighted** for each metric.

These plots help assess **which feature selection techniques yield the best results** for each model.

## Output

Results are stored in:

```
data/reports/{TASK}/
  ├── merged_results.tsv  # Merged accuracy results for all models
  ├── best_results_per_model_rmse.tsv  #Best RMSE values per model
  ├── best_results_per_model_r2.tsv  #Best R² values per model
  ├── best_results_overall_rmse.tsv  #Top 10 models ranked by RMSE
  ├── best_results_overall_r2.tsv  #Top 10 models ranked by $R^2$
  ├── metrics_visualisation_{TASK}.png  # Performance visualization per task
  ├── {model}/{feature_selection}/  # Model-specific results
  │   ├── accuracy.tsv  # Model performance metrics
  │   ├── predictions.tsv  # Predicted metabolites
```


**SuperSOS for completion**

Alignment preprocessing notebook? Check with maiia

**Necessary checks to do:**

* Functionality with Additional visualization
* Some elements of the environment (anndata and scanpy or something else?) contradict one another, for my pc works but for new installations need to check

**Possible additions to be made:**

* VAE preprocessing with scvi
* New models for predictions (GNN, VAE, other)
* Preprocess MSI data
* 

*Alternative to using profile_gpu:*

```
snakemake --jobs 10 --cluster "mkdir -p logs/{rule} && sbatch --partition=gpu_p --gres=gpu:1 --mem=32000 --qos=gpu_normal --job-name=smk-{rule}-{wildcards} --output=logs/{rule}/%j-{rule}-{wildcards}.out --error=logs/{rule}/%j-{rule}-{wildcards}.err --nice=10000 --exclude=supergpu05,supergpu08,supergpu07,supergpu02,supergpu03 --parsable" --cluster-cancel "scancel {cluster_jobid}"
```
