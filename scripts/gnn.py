import numpy as np
import pandas as pd
import scanpy as sc
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import r2_score, mean_squared_error
from scipy.stats import pearsonr, spearmanr
from torch_geometric.nn import GCNConv
from torch_geometric.data import Data

class GNNModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(GNNModel, self).__init__()
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, output_dim)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)
        x = torch.relu(x)
        x = self.conv2(x, edge_index)
        return x

def run_gnn(adata_rna_train,
            adata_rna_test,
            adata_msi_train,
            adata_msi_test,
            params,
            featsel,
            **kwargs):
    # --- Expecting 'edge_index' in kwargs (a torch.LongTensor of shape [2, num_edges]) ---
    if 'edge_index' not in kwargs:
        raise ValueError("GNN requires 'edge_index' in kwargs.")
    edge_index = kwargs['edge_index']

    # --- Feature selection ---
    if featsel in ["hvg", "svd"]:
        X_train = adata_rna_train.X  
        X_test = adata_rna_test.X  
    elif featsel in ["hvg_svd", "hvg_svd_graph", "svd_graph"]:
        X_train = adata_rna_train.obsm.get("svd_features", adata_rna_train.obsm["svd_graph"])
        X_test = adata_rna_test.obsm.get("svd_features", adata_rna_test.obsm["svd_graph"])
    else:
        raise ValueError(f"Unsupported feature selection method: {featsel}")
    Y_train = adata_msi_train.X
    Y_test = adata_msi_test.X

    # --- Convert to torch tensors ---
    X_train = torch.tensor(np.array(X_train), dtype=torch.float32)
    Y_train = torch.tensor(np.array(Y_train), dtype=torch.float32)
    X_test = torch.tensor(np.array(X_test), dtype=torch.float32)
    Y_test = torch.tensor(np.array(Y_test), dtype=torch.float32)

    # --- Create torch_geometric Data objects ---
    data_train = Data(x=X_train, edge_index=edge_index, y=Y_train)
    data_test = Data(x=X_test, edge_index=edge_index, y=Y_test)

    # --- Device Setup: use GPU if available ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_train = data_train.to(device)
    data_test = data_test.to(device)

    # --- Hyperparameters ---
    hidden_dim = int(params.get('hidden_dim', 64))
    lr = float(params.get('lr', 1e-3))
    epochs = int(params.get('epochs', 100))
    input_dim = X_train.shape[1]
    output_dim = Y_train.shape[1]

    # --- Initialize model and optimizer ---
    model = GNNModel(input_dim, hidden_dim, output_dim).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    mse_loss = nn.MSELoss()

    # --- Training loop ---
    model.train()
    for epoch in range(epochs):
        optimizer.zero_grad()
        out = model(data_train)
        loss = mse_loss(out, data_train.y)
        loss.backward()
        optimizer.step()
        print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}")

    # --- Inference ---
    model.eval()
    with torch.no_grad():
        Y_pred = model(data_test).cpu().numpy()

    # --- Evaluation metrics ---
    Y_test_np = Y_test.cpu().numpy()
    pearson_corr = pearsonr(Y_pred.flatten(), Y_test_np.flatten())[0]
    spearman_corr = spearmanr(Y_pred.flatten(), Y_test_np.flatten())[0]
    rmse_test = np.sqrt(mean_squared_error(Y_test_np, Y_pred))
    r2_test = r2_score(Y_test_np, Y_pred)

    results = pd.DataFrame({
        'rmse': [rmse_test],
        'r2': [r2_test],
        'pearson': [pearson_corr],
        'spearman': [spearman_corr]
    })

    return results
