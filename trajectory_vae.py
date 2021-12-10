from tqdm import trange
import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor
from generator import TrajectoryGenerator, Scale
from functools import partial


def tensor_to_array(t):
    return t.detach().cpu().numpy()


def train(X_train, dt, n_latent_dims, n_epochs, beta=1.0, batch_size=16, model_filename="vtae.model", verbose=0):
    """Train VTAE.

    Parameters
    ----------
    X_train : numpy array, shape (n_demonstrations, n_task_dims, n_steps)
        Training data

    dt : float
        Time difference between steps

    n_latent_dims : int
        Number of latent dimensions that we want to reduce the data to

    n_epochs : int
        Number of training epochs

    beta : float, optional (default: 1)
        Hyperparameter of beta-VAE loss

    batch_size : int, optional (default: 16)
        Batch size for backprop

    model_filename : str, optional (default: 'vtae.model')
        Name of the file in which we store the model

    verbose : int, optional (default: 0)
        Verbosity level
    """
    n_demonstrations = X_train.shape[0]
    n_task_dims = X_train.shape[1]
    n_steps = X_train.shape[2]
    X_train = X_train.reshape(n_demonstrations, -1)

    vtae, loss = make_vtae(n_latent_dims, n_task_dims, n_steps, dt, beta=beta)
    train_vae(X_train, vtae, loss, batch_size, n_epochs, model_filename=model_filename, verbose=verbose)


def load_vtae(n_latent_dims, n_task_dims, n_steps, dt, dropout=None, filename="vtae.model"):
    """Load VTAE from a file.

    Parameters
    ----------
    n_latent_dims : int
        Number of latent dimensions that we want to reduce the data to

    n_task_dims : int
        Number of dimensions of the trajectory / task space

    n_steps : int
        Number of steps in each trajectory

    dt : float
        Time difference between steps

    dropout : float, optional (default: None)
        Dropout percentage for decoder last layer, deactivated by default

    filename : str, optional (default: 'vtae.model')
        Name of the file in which we store the model

    Returns
    -------
    model : VTAE
        VTAE model loaded from file
    """
    vtae = VTAE(n_latent_dims, n_task_dims, n_steps, dt, dropout)
    if torch.cuda.is_available():
        vtae.cuda()
        map_location = None
    else:
        map_location = lambda storage, location: storage
    vtae.load_state_dict(torch.load(filename, map_location))
    vtae.eval()
    return vtae


def make_vtae(n_latent_dims, n_task_dims, n_steps, dt, beta=1.0):
    """Create variational trajectory autoencoder.

    Move the model to GPU if cuda is available.

    Parameters
    ----------
    n_latent_dims : int
        Number of latent dimensions that we want to reduce the data to

    n_task_dims : int
        Number of dimensions of the trajectory / task space

    n_steps : int
        Number of steps in each trajectory

    dt : float
        Time difference between steps

    beta : float, optional (default: 1)
        Hyperparameter of beta-VAE loss

    Returns
    -------
    model : VTAE
        New model, untrained

    loss : callable
        Loss function
    """
    model = VTAE(n_latent_dims, n_task_dims, n_steps, dt)
    if torch.cuda.is_available():
        model.cuda()
    return model, partial(beta_vae_loss, beta=beta)


class VTAE(nn.Module):
    """Variational trajectory autoencoder.

    Parameters
    ----------
    n_latent_dims : int
        Number of latent dimensions that we want to reduce the data to

    n_task_dims : int
        Number of dimensions of the trajectory / task space

    n_steps : int
        Number of steps in each trajectory

    dt : float
        Time difference between steps

    dropout : float, optional (default: None)
        Dropout percentage for decoder last layer, deactivated by default
    """
    def __init__(self, n_latent_dims, n_task_dims, n_steps, dt, dropout=None):
        super(VTAE, self).__init__()
        n_features = n_task_dims * n_steps
        self.encoder = nn.Sequential(
            nn.Linear(n_features, n_features // 2),
            Scale(0.1),
            nn.LeakyReLU(inplace=True),
            nn.Linear(n_features // 2, n_features // 4),
            Scale(0.1),
            nn.LeakyReLU(inplace=True),
            nn.Linear(n_features // 4, n_features // 8),
            Scale(0.1),
            nn.LeakyReLU(inplace=True),
        )
        self.mean_encoder = nn.Linear(n_features // 8, n_latent_dims)
        self.logvar_encoder = nn.Linear(n_features // 8, n_latent_dims)

        self.trajectory_generator = TrajectoryGenerator(
            n_latent_dims, n_task_dims, n_steps, dt, dropout=dropout)

    def encode(self, x):
        x = self.encoder(x)
        return self.mean_encoder(x), self.logvar_encoder(x)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps.mul(std).add_(mu)

    def decode(self, z):
        return self.trajectory_generator(z)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        y = self.trajectory_generator(z)
        return y, mu, logvar


def beta_vae_loss(recon_x, x, mu, logvar, beta=1.0):
    SSE = F.mse_loss(recon_x, x, reduction="sum")
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return SSE + beta * KLD


def train_vae(X, model, loss, batch_size, n_epochs, model_filename="vtae.model", verbose=0):
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, betas=(0.5, 0.999))
    checkpoint_saver = CheckpointSaver(filename=model_filename)
    pbar = trange(n_epochs)
    n_train_samples = len(X)

    for _ in pbar:
        for i in range(n_train_samples // batch_size):
            batch = Tensor(X[i * batch_size:(i + 1) * batch_size])

            reconstruction, mu, logvar = model(batch)
            l = loss(reconstruction, batch, mu, logvar)

            opt.zero_grad()
            l.backward()
            opt.step()

            train_loss = l.item()
            checkpoint_saver(train_loss, model)

            if verbose:
                pbar.set_description("Error: %.3f; Epochs" % train_loss)


class CheckpointSaver:
    """Saves model when loss decreases."""
    def __init__(self, filename):
        self.best_loss = None
        self.filename = filename

    def __call__(self, loss, model):
        if self.best_loss is None:
            self.best_loss = loss
            self.save_checkpoint(model)
        elif loss < self.best_loss:
            self.best_loss = loss
            self.save_checkpoint(model)

    def save_checkpoint(self, model):
        torch.save(model.state_dict(), self.filename)

