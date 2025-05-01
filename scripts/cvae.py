import numpy as np
import pandas as pd
import scanpy as sc
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import r2_score, mean_squared_error
from scipy.stats import pearsonr, spearmanr
from scipy.sparse import issparse
from sklearn.metrics import mean_absolute_error
import random

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

    # Set the random seed directly inside the function to ensure reproducibility
    seed = 666
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)  # For multi-GPU
        torch.backends.cudnn.deterministic = True  # Ensures deterministic results on GPU
        torch.backends.cudnn.benchmark = False  # Ensures determinism, disables optimization for performance
    
    # --- Feature selection ---
    # Select features based on the provided feature selection method
    if featsel in {"hvg", "hvg_nomsi", "none"}:
        X_train = adata_rna_train.X  
        X_test = adata_rna_test.X  
        Y_train, Y_test = adata_msi_train.X, adata_msi_test.X
    elif featsel in {"hvg_svd", "svd", "svd_hvgmsi"}:
        X_train = adata_rna_train.obsm["svd_features"]
        X_test = adata_rna_test.obsm["svd_features"]
        Y_train, Y_test = adata_msi_train.X, adata_msi_test.X
    elif featsel in {"hvg_svd_graph", "svd_graph", "svd_graph_hvgmsi"}:
        X_train = adata_rna_train.obsm["svd_graph"]
        X_test = adata_rna_test.obsm["svd_graph"] 
        Y_train, Y_test = adata_msi_train.X, adata_msi_test.X
    elif featsel == "hvg_rna":
        X_train = adata_rna_train.X  
        X_test = adata_rna_test.X  
        Y_train, Y_test = adata_msi_train.obsm["X_pca_split"], adata_msi_test.obsm["X_pca_split"]
    elif featsel == "hvg_rna_svd":
        X_train = adata_rna_train.obsm["svd_features"]
        X_test = adata_rna_test.obsm["svd_features"]
        Y_train, Y_test = adata_msi_train.obsm["X_pca_split"], adata_msi_test.obsm["X_pca_split"]
    elif featsel == "hvg_rna_svd_graph":
        X_train = adata_rna_train.obsm["svd_graph"]
        X_test = adata_rna_test.obsm["svd_graph"] 
        Y_train, Y_test = adata_msi_train.obsm["X_pca_split"], adata_msi_test.obsm["X_pca_split"]
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
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, worker_init_fn=np.random.seed(seed))

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
        # Train prediction (reconstruct from encoded latent space)
        y_recon_train, _, _ = model(X_train, Y_train)
        Y_pred_train = y_recon_train.cpu().numpy()


    # --- Evaluation metrics ---
    Y_test_np = Y_test.cpu().numpy()
    Y_train_np = Y_train.cpu().numpy()

    # Global correlation
    pearson_corr = pearsonr(Y_pred.flatten(), Y_test_np.flatten())[0]
    spearman_corr = spearmanr(Y_pred.flatten(), Y_test_np.flatten())[0]

    # Per-metabolite correlation
    per_met_pearsons = [pearsonr(Y_pred[:, i], Y_test_np[:, i])[0] for i in range(Y_test_np.shape[1])]
    per_met_spearmans = [spearmanr(Y_pred[:, i], Y_test_np[:, i])[0] for i in range(Y_test_np.shape[1])]
    avg_pearson_per_metabolite = np.nanmean(per_met_pearsons)
    avg_spearman_per_metabolite = np.nanmean(per_met_spearmans)

    # Standard metrics
    rmse_test = np.sqrt(mean_squared_error(Y_test_np, Y_pred))
    r2_test = r2_score(Y_test_np, Y_pred)
    mae_test = mean_absolute_error(Y_test_np, Y_pred)
    

    # Relative RMSE (skip zero-mean metabolites)
    per_met_rmse = [np.sqrt(np.mean((Y_pred[:, i] - Y_test_np[:, i]) ** 2)) for i in range(Y_test_np.shape[1])]
    per_met_mean = [np.mean(Y_test_np[:, i]) for i in range(Y_test_np.shape[1])]
    rel_rmse = [r / m if m != 0 else np.nan for r, m in zip(per_met_rmse, per_met_mean)]
    avg_rel_rmse = np.nanmean(rel_rmse)

    # Train set metrics
    rmse_train = np.sqrt(mean_squared_error(Y_train_np, Y_pred_train))
    r2_train = r2_score(Y_train_np, Y_pred_train)
    mae_train = mean_absolute_error(Y_train_np, Y_pred_train)


    # Save metrics
    metrics = pd.DataFrame({
        'rmse': [rmse_test],
        'mae': [mae_test],
        'r2': [r2_test],
        'pearson': [pearson_corr],
        'spearman': [spearman_corr],
        'avg_pearson_per_metabolite': [avg_pearson_per_metabolite],
        'avg_spearman_per_metabolite': [avg_spearman_per_metabolite],
        'avg_rel_rmse': [avg_rel_rmse],
        'train_rmse': [rmse_train],
        'train_mae': [mae_train],
        'train_r2': [r2_train]

    })

    #Add this for interpretability later, check outputs of each model's preds
    predictions = pd.DataFrame({
        'y_true': Y_test_np.flatten(),
        'y_pred': Y_pred.flatten()
        })

    return metrics, predictions

