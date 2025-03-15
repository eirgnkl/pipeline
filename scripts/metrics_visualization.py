import os
import pandas as pd
import matplotlib.pyplot as plt

# Retrieve parameters from Snakemake
num_top_models = int(snakemake.params.top_model)  # Number of top models to display
selected_task = snakemake.wildcards.task           # Task name

# Define file paths
file_path = os.path.join("data/reports", selected_task, "merged_results.tsv")
save_directory = os.path.join("data/reports", selected_task)
filename = f"metrics_visualisation_{selected_task}.png"
full_path = os.path.join(save_directory, filename)

# Ensure save directory exists
os.makedirs(save_directory, exist_ok=True)

# Load the data for the selected task
df = pd.read_csv(file_path, sep='\t')

# Define metrics and their optimization direction
metrics = ['r2', 'mae', 'rmse', 'pearson', 'spearman']
best_direction = {'r2': 'max', 'mae': 'min', 'rmse': 'min', 'pearson': 'max', 'spearman': 'max'}
highlight_colors = {'r2': 'steelblue', 'mae': 'forestgreen', 'rmse': 'darkorange', 'pearson': 'purple', 'spearman': 'red'}
default_color = 'lightgray'
text_color = 'white'

# Mapping featsel to a short label for the x-axis
featsel_map = {
    'hvg': 'h',
    'hvg_svd': 'hs',
    'hvg_svd_graph': 'hsg',
    'svd': 's',
    'svd_graph': 'sg'
}

def get_method_label(method_name, featsel, method_params):
    """
    Returns a multi-line label for the x-axis in the form:
    
      method_name
      key1=val1
      key2=val2
      ...
      featsel_short
    
    If no method_params are provided, the label consists of just the method_name
    on the first line and the shortened featsel on the second line.
    """
    # Use the short featsel for the x-axis label
    featsel_short = featsel_map.get(featsel, featsel)
    
    # Convert method_params from string to dict if needed
    if isinstance(method_params, str):
        method_params = eval(method_params)
        
    if not method_params or not isinstance(method_params, dict):
        return f"{method_name}\n{featsel_short}"
    
    # Create a list of parameter strings and keep only the first two pairs
    param_parts = [f"{k}={v}" for k, v in method_params.items()]
    param_parts = param_parts[:2]
    param_str = "\n".join(param_parts)
    
    # Return the multi-line label
    return f"{method_name}\n{param_str}\n{featsel_short}"

# Prepare a list of columns to select
columns_to_select = ['method_name', 'featsel']
if 'method_params' in df.columns:
    columns_to_select.append('method_params')

# Create a dictionary to store top performers for each metric
top_performers = {}
for metric in metrics:
    if metric in df.columns:
        if best_direction[metric] == 'max':
            top_indices = df[metric].nlargest(num_top_models).index
        else:
            top_indices = df[metric].nsmallest(num_top_models).index
        selected_columns = columns_to_select + [metric]
        top_performers[metric] = df.loc[top_indices, selected_columns]

# Create a subplot for each metric
num_metrics = len(metrics)
fig, axes = plt.subplots(1, num_metrics, figsize=(6 * num_metrics, 6), squeeze=False)
fig.suptitle(f"Performance Comparison for Task: {selected_task}", fontsize=14, fontweight='bold')

for i, metric in enumerate(metrics):
    ax = axes[0, i]
    if metric not in top_performers:
        continue

    top_df = top_performers[metric]
    method_labels = []
    original_featsel = []  # to annotate inside bars
    for _, row in top_df.iterrows():
        featsel_val = row['featsel']
        original_featsel.append(featsel_val)
        label = get_method_label(
            row['method_name'],
            featsel_val,
            row['method_params'] if 'method_params' in top_df.columns and pd.notna(row['method_params']) else None
        )
        method_labels.append(label)
    
    y_values = top_df[metric].values

    # Determine best value for highlighting
    if best_direction[metric] == 'max':
        overall_best_value = max(y_values)
    else:
        overall_best_value = min(y_values)
    colors = [highlight_colors[metric] if val == overall_best_value else default_color for val in y_values]

    bars = ax.bar(method_labels, y_values, color=colors)

    # Annotate metric value above each bar
    for bar, value in zip(bars, y_values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            value + (ax.get_ylim()[1] * 0.002),
            f"{value:.3f}",
            ha='center', va='bottom', fontsize=10, color='black'
        )
    # Annotate the original featsel inside each bar vertically
    for bar, featsel in zip(bars, original_featsel):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() / 2,
            featsel,
            ha='center', va='center', fontsize=8, color=text_color, rotation=90
        )

    ax.set_title(f"Top {num_top_models} {metric.upper()} Performers", fontsize=10)
    ax.set_ylabel(metric.upper())
    ax.set_xticks([bar.get_x() + bar.get_width() / 2 for bar in bars])
    ax.set_xticklabels(method_labels, rotation=45, ha='right', fontsize=8)

plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.savefig(full_path, dpi=300, bbox_inches='tight')
print(f"Saved plot for {selected_task}: {full_path}")
plt.show()
