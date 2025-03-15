import numpy as np
import pandas as pd
import scanpy as sc
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from scipy.stats import pearsonr, spearmanr
from scipy.sparse import issparse

def convert_to_tensor(X):
    """Convert a sparse matrix or other format to a torch tensor."""
    if issparse(X):
        X = X.toarray()
    if not isinstance(X, np.ndarray):
        if isinstance(X, torch.Tensor):
            X = X.cpu().numpy()
        else:
            X = np.array(X)
    return torch.tensor(X, dtype=torch.float32)

class MultiLayerGCN(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=3, dropout=0.5):
        super(MultiLayerGCN, self).__init__()
        # First GCN layer
        self.layers = torch.nn.ModuleList()
        self.layers.append(GCNConv(input_dim, hidden_dim))
        # Middle layers (if any)
        for _ in range(num_layers - 2):
            self.layers.append(GCNConv(hidden_dim, hidden_dim))
        # Output layer
        self.output_layer = GCNConv(hidden_dim, output_dim)
        self.dropout = dropout

    def forward(self, x, edge_index):
        # Pass data through each GCN layer with ReLU and dropout
        for layer in self.layers:
            x = layer(x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        # Final output layer
        x = self.output_layer(x, edge_index)
        return x

def run_gnn(adata_rna_train,
            adata_rna_test,
            adata_msi_train,
            adata_msi_test,
            params,
            featsel,
            **kwargs):
    # --- Feature Selection ---
    if featsel == "hvg":
        X_train_np = adata_rna_train.X  
        X_test_np = adata_rna_test.X  
        Y_train_np, Y_test_np = adata_msi_train.X, adata_msi_test.X
    elif featsel == "hvg_svd":
        X_train_np = adata_rna_train.obsm["svd_features"]
        X_test_np = adata_rna_test.obsm["svd_features"]
        Y_train_np, Y_test_np = adata_msi_train.X, adata_msi_test.X
    elif featsel == "hvg_svd_graph":
        X_train_np = adata_rna_train.obsm["svd_graph"]
        X_test_np = adata_rna_test.obsm["svd_graph"] 
        Y_train_np, Y_test_np = adata_msi_train.X, adata_msi_test.X
    elif featsel == "svd":
        X_train_np = adata_rna_train.obsm["svd_features"]
        X_test_np = adata_rna_test.obsm["svd_features"]
        Y_train_np, Y_test_np = adata_msi_train.X, adata_msi_test.X
    elif featsel == "svd_graph":
        X_train_np = adata_rna_train.obsm["svd_graph"]
        X_test_np = adata_rna_test.obsm["svd_graph"]
        Y_train_np, Y_test_np = adata_msi_train.X, adata_msi_test.X
    else:
        raise ValueError(f"Unsupported feature selection method: {featsel}")
    
    # --- Spatial Coordinates ---
    # Here we assume the spatial coordinates are stored under 'spatial_warp' in the .obsm slot.
    coords_rna_train = adata_rna_train.obsm["spatial_warp"]
    coords_rna_test = adata_rna_test.obsm["spatial_warp"]

    # --- Convert Data to Tensors ---
    X_train_tensor = convert_to_tensor(X_train_np)
    Y_train_tensor = convert_to_tensor(Y_train_np)
    X_test_tensor  = convert_to_tensor(X_test_np)
    Y_test_tensor  = convert_to_tensor(Y_test_np)

    # --- Device Setup ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    X_train_tensor = X_train_tensor.to(device)
    Y_train_tensor = Y_train_tensor.to(device)
    X_test_tensor  = X_test_tensor.to(device)
    Y_test_tensor  = Y_test_tensor.to(device)

    # --- Hyperparameters ---
    hidden_dim = int(params.get('hidden_dim', 256))
    lr = float(params.get('lr', 0.001))
    num_layers = int(params.get('layers', 3))
    dropout = float(params.get('dropout', 0.3))
    epochs = int(params.get('epochs', 2000))
    k_train = int(params.get('k_neighbors', 15))
    k_test = int(params.get('k_test_neighbors', 5))

    input_dim = X_train_tensor.shape[1]
    output_dim = Y_train_tensor.shape[1]

    # --- Build Training Graph ---
    knn_train = NearestNeighbors(n_neighbors=k_train).fit(coords_rna_train)
    _, indices_train = knn_train.kneighbors(coords_rna_train)
    train_edges = []
    for i, neighbors in enumerate(indices_train):
        for neighbor in neighbors:
            if i != neighbor:  # Avoid self-loops
                train_edges.append([i, neighbor])
    train_edge_index = torch.tensor(train_edges, dtype=torch.long).t().contiguous().to(device)
    train_data = Data(x=X_train_tensor, edge_index=train_edge_index)

    # --- Initialize GCN Model ---
    model = MultiLayerGCN(input_dim, hidden_dim, output_dim, num_layers=num_layers, dropout=dropout).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = torch.nn.MSELoss()

    # --- Train GCN Model ---
    model.train()
    for epoch in range(epochs):
        optimizer.zero_grad()
        out = model(train_data.x, train_data.edge_index)
        loss = criterion(out, Y_train_tensor)
        loss.backward()
        optimizer.step()
        if epoch % 1000 == 0:
            print(f"Epoch {epoch}, Loss: {loss.item()}")

    # --- Build Test Graph ---
    knn_test = NearestNeighbors(n_neighbors=k_test).fit(coords_rna_test)
    _, indices_test = knn_test.kneighbors(coords_rna_test)
    test_edges = []
    for i, neighbors in enumerate(indices_test):
        for neighbor in neighbors:
            if i != neighbor:
                test_edges.append([i, neighbor])
    test_edge_index = torch.tensor(test_edges, dtype=torch.long).t().contiguous().to(device)
    test_data = Data(x=X_test_tensor, edge_index=test_edge_index)

    # --- Evaluate the Model ---
    model.eval()
    with torch.no_grad():
        Y_pred_train = model(train_data.x, train_data.edge_index).detach().cpu().numpy()
        Y_pred_test = model(test_data.x, test_data.edge_index).detach().cpu().numpy()

    # --- Compute Evaluation Metrics (Test Set) ---
    Y_test_np = Y_test_tensor.cpu().numpy()
    pearson_corr = pearsonr(Y_pred_test.flatten(), Y_test_np.flatten())[0]
    spearman_corr = spearmanr(Y_pred_test.flatten(), Y_test_np.flatten())[0]
    rmse_test = np.sqrt(mean_squared_error(Y_test_np, Y_pred_test))
    r2_test = r2_score(Y_test_np, Y_pred_test)
    mae_test = mean_absolute_error(Y_test_np, Y_pred_test)

    metrics = pd.DataFrame({
        'rmse': [rmse_test],
        'mae': [mae_test],
        'r2': [r2_test],
        'pearson': [pearson_corr],
        'spearman': [spearman_corr]
    })

    predictions = pd.DataFrame({
        'y_true': Y_test_np.flatten(),
        'y_pred': Y_pred_test.flatten()
    })

    return metrics, predictions
