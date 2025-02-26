import numpy as np
import pandas as pd
import scanpy as sc
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import r2_score, mean_squared_error
from scipy.stats import pearsonr, spearmanr

def gaussian_nll(y, pi, mu, sigma):
    # y: (batch, output_dim)
    # pi: (batch, K) mixture weights (softmaxed)
    # mu: (batch, K, output_dim)
    # sigma: (batch, K, output_dim), positive
    m = torch.distributions.Normal(mu, sigma)
    log_prob = m.log_prob(y.unsqueeze(1).expand_as(mu))  # (batch, K, output_dim)
    log_prob = torch.sum(log_prob, dim=2)  # sum over output_dim -> (batch, K)
    weighted_log_prob = log_prob + torch.log(pi + 1e-8)
    log_sum = torch.logsumexp(weighted_log_prob, dim=1)
    return -torch.mean(log_sum)

class MDN(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim, num_components):
        super(MDN, self).__init__()
        self.hidden = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
        )
        self.pi = nn.Linear(hidden_dim, num_components)
        self.mu = nn.Linear(hidden_dim, num_components * output_dim)
        self.sigma = nn.Linear(hidden_dim, num_components * output_dim)
        self.num_components = num_components
        self.output_dim = output_dim

    def forward(self, x):
        h = self.hidden(x)
        pi = torch.softmax(self.pi(h), dim=1)  # (batch, K)
        mu = self.mu(h).view(-1, self.num_components, self.output_dim)
        sigma = torch.exp(self.sigma(h).view(-1, self.num_components, self.output_dim))
        return pi, mu, sigma

def run_mdn(adata_rna_train,
            adata_rna_test,
            adata_msi_train,
            adata_msi_test,
            params,
            featsel,
            **kwargs):
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
    X_test = torch.tensor(np.array(X_test), dtype=torch.float32)
    Y_train = torch.tensor(np.array(Y_train), dtype=torch.float32)
    Y_test = torch.tensor(np.array(Y_test), dtype=torch.float32)

    # --- Hyperparameters ---
    hidden_dim = int(params.get('hidden_dim', 128))
    num_components = int(params.get('num_components', 3))
    lr = float(params.get('lr', 1e-3))
    epochs = int(params.get('epochs', 100))
    batch_size = int(params.get('batch_size', 32))

    input_dim = X_train.shape[1]
    output_dim = Y_train.shape[1]

    model = MDN(input_dim, output_dim, hidden_dim, num_components)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    dataset = torch.utils.data.TensorDataset(X_train, Y_train)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
    model.train()
    for epoch in range(epochs):
        epoch_loss = 0
        for x_batch, y_batch in dataloader:
            optimizer.zero_grad()
            pi, mu, sigma = model(x_batch)
            loss = gaussian_nll(y_batch, pi, mu, sigma)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        # Optional: print training progress
        # print(f"Epoch {epoch+1}/{epochs}, Loss: {epoch_loss/len(dataloader):.4f}")

    # --- Inference ---
    model.eval()
    with torch.no_grad():
        pi, mu, sigma = model(X_test)
        # Compute expected value: weighted average of the means
        Y_pred = (pi.unsqueeze(2) * mu).sum(dim=1).numpy()

    # --- Evaluation metrics ---
    pearson_corr = pearsonr(Y_pred.flatten(), Y_test.numpy().flatten())[0]
    spearman_corr = spearmanr(Y_pred.flatten(), Y_test.numpy().flatten())[0]
    rmse_test = np.sqrt(mean_squared_error(Y_test.numpy(), Y_pred))
    r2_test = r2_score(Y_test.numpy(), Y_pred)

    results = pd.DataFrame({
        'rmse': [rmse_test],
        'r2': [r2_test],
        'pearson': [pearson_corr],
        'spearman': [spearman_corr]
    })

    return results
