## Pipeline for Metabolite Prediction from Gene Expression

This repository contains a Snakemake pipeline for predicting **metabolic distribution from gene expression data** using several **machine learning models** and **feature selection techniques**.

The goal is to evaluate whether **consecutive slides** can perform as well as **same-slide** data in predicting metabolites.

## Pipeline Structure

The pipeline is structured as follows:

1. **Feature Preprocessing**: Selects relevant features using different feature selection methods.
2. **Model Training**: Runs Ridge, Lasso, Linear Regression, XGBoost and a CVAE.
3. **Evaluation**: Assesses performance using 4 different metrics (RMSE, Pearson and Spearmann Correlation and $R^2$).
4. **Comparison of Consecutive vs. Same-Slide Data**.

## Feature Preprocessing

The dataset undergoes multiple feature selection techniques to reduce dimensionality and improve model performance. The available methods are:

- **hvg**: High-variance genes.
- **hvg+svd**: High-variance genes + singular value decomposition.
- **hvg+svd+graph**: Adds graph-based selection.
- **svd**: Singular value decomposition.
- **svd+graph**: Graph-based selection on SVD processed data.

### How to set Tasks, Feature Selection and Models:

Feature selection is specified in `cof.yaml` under the `featsel` key.

1. Name your TASK and set correct paths to your RNA and MSI datasets
2. Define the name of the split that you want to use for the task (this is used later in both feature selection and training thus it's set up in the config.yaml)
3. Specify the methods (models) that you want to use to make predictions. Set the parameter combination that each method is run with in the params/[method]_params.tsv.
4. Select the different ways of preprocessing that you want your models to run with.

## Models Implemented

The pipeline supports the following regression models:

- **Ridge Regression**: Handles multicollinearity by adding an L2 penalty.
- **Lasso Regression**: Adds an L1 penalty for feature selection.
- **Linear Regression**: Standard least squares regression.
- **XGBoost**: Gradient boosting for non-linear patterns.

### How to Select Model

Specify the model in `config.yaml` under `model`.

Example:

```yaml
model: ridge
```

---

## Hyperparameters to Tune

Each model has parameters that users can configure in `params.tsv`.

### Ridge & Lasso

```yaml
alpha: 1.0  # Regularization strength
```

### XGBoost

```yaml
learning_rate: 0.1
n_estimators: 100
max_depth: 3
```

---

## Running the Pipeline

To execute the pipeline, use:

```bash
snakemake --cores <num_cores> --profile profile_gpu
```

For dry-run mode:

```bash
snakemake -n
```

---

## Output

Results are stored in:

```
data/reports/{TASK}/  # Best results for each task
  ├── best_results.tsv  # Summary of best-performing models
  ├── {model}/{feature_selection}/  # Model-specific results
  │   ├── accuracy.tsv  # Model performance metrics
  │   ├── predictions.tsv  # Predicted metabolites

```

.


**SuperSOS for completion**

Alignment preprocessing notebook? Check with maiia

**Necessary checks to do:**

* Functionality of pipeline with 1) all tasks 2) each task alone
* Functionality of pipeline with elastic net and new extra models
* Functionality with Additional visualization
* Some elements of the environment (anndata and scanpy or something else?) contradict one another, for my pc works but for new installations need to check

**Possible additions to be made:**

* VAE preprocessing with scvi
* New models for predictions (GNN, VAE, other)

*Alternative to using profile_gpu:*

```
snakemake --jobs 10 --cluster "mkdir -p logs/{rule} && sbatch --partition=gpu_p --gres=gpu:1 --mem=32000 --qos=gpu_normal --job-name=smk-{rule}-{wildcards} --output=logs/{rule}/%j-{rule}-{wildcards}.out --error=logs/{rule}/%j-{rule}-{wildcards}.err --nice=10000 --exclude=supergpu05,supergpu08,supergpu07,supergpu02,supergpu03 --parsable" --cluster-cancel "scancel {cluster_jobid}"
```
