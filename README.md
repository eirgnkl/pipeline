Creating a functional pipeline for automation of metabolomics predictions.

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

*WARNING:* profile_gpu is not functioning currently, if you want to use it call:

```
snakemake --jobs 10 --cluster "mkdir -p logs/{rule} && sbatch --partition=gpu_p --gres=gpu:2 --mem=32000 --qos=gpu_normal --job-name=smk-{rule}-{wildcards} --output=logs/{rule}/%j-{rule}-{wildcards}.out --error=logs/{rule}/%j-{rule}-{wildcards}.err --nice=10000 --exclude=supergpu05,supergpu08,supergpu07,supergpu02,supergpu03 --parsable" --cluster-cancel "scancel {cluster_jobid}"
```
