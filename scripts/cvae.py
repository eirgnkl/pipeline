import numpy as np
import pandas as pd
import scanpy as sc
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import r2_score, mean_squared_error
from scipy.stats import pearsonr, spearmanr
from scipy.sparse import issparse

def convert_to_tensor(X):
    # If X is sparse, convert to dense
    if issparse(X):
        X = X.toarray()
    # If it's not a numpy array, try to convert it to one
    if not isinstance(X, np.ndarray):
        # If it's already a torch tensor, move it to CPU and convert to numpy
        if isinstance(X, torch.Tensor):
            X = X.cpu().numpy()
        else:
            X = np.array(X)
    return torch.tensor(X, dtype=torch.float32)

class CVAE(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim, latent_dim):
        super(CVAE, self).__init__()
        # Encoder: takes concatenated (x, y)
        self.encoder = nn.Sequential(
            nn.Linear(input_dim + output_dim, hidden_dim),
            nn.ReLU(),
        )
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)
        # Decoder: takes concatenated (x, z)
        self.decoder = nn.Sequential(
            nn.Linear(input_dim + latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)  # inherits device from std
        return mu + eps * std

    def forward(self, x, y):
        # Concatenate condition (x) and target (y) for encoding
        encoder_input = torch.cat([x, y], dim=1)
        h = self.encoder(encoder_input)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        z = self.reparameterize(mu, logvar)
        # Decode using condition x and latent variable z
        decoder_input = torch.cat([x, z], dim=1)
        y_recon = self.decoder(decoder_input)
        return y_recon, mu, logvar

def run_cvae(adata_rna_train,
             adata_rna_test,
             adata_msi_train,
             adata_msi_test,
             params,
             featsel,
             **kwargs):
    
    
    # --- Feature selection ---
    # Select features based on the provided feature selection method
    if featsel == "hvg":
        X_train = adata_rna_train.X  
        X_test = adata_rna_test.X  
        Y_train, Y_test = adata_msi_train.X, adata_msi_test.X
    elif featsel == "hvg_svd":
        X_train = adata_rna_train.obsm["svd_features"]
        X_test = adata_rna_test.obsm["svd_features"]
        Y_train, Y_test = adata_msi_train.X, adata_msi_test.X
    elif featsel == "hvg_svd_graph":
        X_train = adata_rna_train.obsm["svd_graph"]
        X_test = adata_rna_test.obsm["svd_graph"] 
        Y_train, Y_test = adata_msi_train.X, adata_msi_test.X
    elif featsel == "svd":
        X_train = adata_rna_train.obsm["svd_features"]
        X_test = adata_rna_test.obsm["svd_features"]
        Y_train, Y_test = adata_msi_train.X, adata_msi_test.X
    elif featsel == "svd_graph":
        X_train = adata_rna_train.obsm["svd_graph"]
        X_test = adata_rna_test.obsm["svd_graph"]
        Y_train, Y_test = adata_msi_train.X, adata_msi_test.X
    else:
        raise ValueError(f"Unsupported feature selection method: {featsel}")

    # --- Convert to torch tensors if necessary ---
    X_train = convert_to_tensor(X_train)
    X_test = convert_to_tensor(X_test)
    Y_train = convert_to_tensor(Y_train)
    Y_test = convert_to_tensor(Y_test)

    # --- Device Setup: use GPU if available ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    X_train = X_train.to(device)
    Y_train = Y_train.to(device)
    X_test = X_test.to(device)
    Y_test = Y_test.to(device)

    # --- Hyperparameters ---
    hidden_dim = int(params.get('hidden_dim', 128))
    latent_dim = int(params.get('latent_dim', 10))
    lr = float(params.get('lr', 1e-3))
    epochs = int(params.get('epochs', 100))
    batch_size = int(params.get('batch_size', 32))

    input_dim = X_train.shape[1]
    output_dim = Y_train.shape[1]

    # --- Initialize model and optimizer ---
    model = CVAE(input_dim, output_dim, hidden_dim, latent_dim).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    mse_loss = nn.MSELoss()

    # --- Training loop ---
    model.train()
    dataset = torch.utils.data.TensorDataset(X_train, Y_train)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

    for epoch in range(epochs):
        epoch_loss = 0.0
        num_batches = 0
        for x_batch, y_batch in dataloader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            y_recon, mu, logvar = model(x_batch, y_batch)
            # Reconstruction loss
            recon_loss = mse_loss(y_recon, y_batch)
            # KL divergence loss
            kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
            loss = recon_loss + kl_loss
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            num_batches += 1

        avg_loss = epoch_loss / num_batches
        print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")

        # --- Inference ---
        model.eval()
        with torch.no_grad():
            z_sample = torch.randn(X_test.size(0), latent_dim).to(device)
            decoder_input = torch.cat([X_test, z_sample], dim=1)
            Y_pred = model.decoder(decoder_input).cpu().numpy()

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
