import numpy as np
import torch
from torch.autograd import Variable
from torch.nn import functional as F
Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor


class TrajectoryGenerator(torch.nn.Module):
    """Generator network for trajectories.

    This can be used, e.g., as a decoder in an autoencoder.

    Parameters
    ----------
    n_latent_dims : int
        Number of dimensions in the latent space

    n_task_dims : int
        Number of dimensions of the space in which the trajectories will
        be generated

    n_steps : int
        Number of steps in each trajectory

    dt : float
        Time between two successive steps

    initial_scaling : float, optional (default: 0.1)
        Initial scalar scaling of linear layers

    dropout : float, optional (default: None)
        Dropout percentage for last layer, deactivated by default
    """
    def __init__(self, n_latent_dims, n_task_dims, n_steps, dt,
                 initial_scaling=0.1, dropout=None):
        super(TrajectoryGenerator, self).__init__()
        self.n_latent_dims = n_latent_dims
        self.dropout = dropout
        n_features = n_task_dims * n_steps
        self.eps_decoder = torch.nn.Sequential(
            torch.nn.Linear(n_latent_dims, n_features // 8),
            Scale(initial_scaling),
            torch.nn.LeakyReLU(inplace=True),
            torch.nn.Linear(n_features // 8, n_features // 4),
            Scale(initial_scaling),
            torch.nn.LeakyReLU(inplace=True),
            torch.nn.Linear(n_features // 4, n_features),
            Scale(1.0 / dt),
        )
        self.start_decoder = torch.nn.Sequential(
            torch.nn.Linear(n_latent_dims, n_task_dims),
            Scale(initial_scaling),
            torch.nn.LeakyReLU(inplace=True),
            torch.nn.Linear(n_task_dims, n_task_dims),
            Scale(initial_scaling),
            torch.nn.LeakyReLU(inplace=True),
            torch.nn.Linear(n_task_dims, n_task_dims),
        )
        self.trajectory = TrajectoryLayer(n_task_dims, n_steps, dt)

    def forward(self, z):
        eps = self.eps_decoder(z)
        start = self.start_decoder(z)
        if self.dropout is not None:
            eps = F.dropout(eps, self.dropout, training=self.training)
        return self.trajectory(eps, start)


class TrajectoryLayer(torch.nn.Module):
    """A new type of layer that enforces smooth trajectories.

    Parameters
    ----------
    n_task_dims : int
        Number of dimensions of the trajectory / task space

    n_steps : int
        Number of steps in each trajectory

    dt : float
        Time difference between steps
    """
    def __init__(self, n_task_dims, n_steps, dt):
        super(TrajectoryLayer, self).__init__()
        self.n_task_dims = n_task_dims
        self.n_steps = n_steps
        self.dt = dt
        self.L = Variable(Tensor(acceleration_L(n_task_dims, n_steps, dt)))

    def forward(self, eps, start):
        n_samples = eps.shape[0]
        start = start.view(
            n_samples, self.n_task_dims, 1).repeat(1, 1, self.n_steps).view(
            n_samples, self.n_task_dims * self.n_steps)
        return torch.mm(eps, self.L.t()) + start


class Scale(torch.nn.Module):
    """Learnable scalar scaling factor."""
    def __init__(self, init_value=1.0):
        super(Scale, self).__init__()
        self.scale = torch.nn.Parameter(Tensor([init_value]))

    def forward(self, x):
        return x * self.scale


def acceleration_L(n_task_dims, n_steps, dt=1.0):
    """Computes the Cholesky decomposition of a covariance for smooth trajectories."""
    # This finite difference matrix only works for 1D trajectories.
    A = create_fd_matrix_1d(2 * n_steps, dt, order=2)
    covariance = np.linalg.inv(A.T.dot(A))

    # We created A with twice the size that we need and now cut it to the size
    # that we need. We have to do this to ensure that trajectories can diverge
    # from 0 in the end.
    L_per_dim = np.linalg.cholesky(covariance)[:n_steps, :n_steps]

    # We copy L for each dimension.
    L = np.zeros((n_task_dims * n_steps, n_task_dims * n_steps))
    for d in range(n_task_dims):
        L[d * n_steps:(d + 1) * n_steps, d * n_steps: (d + 1) * n_steps] = L_per_dim
    return L


def create_fd_matrix_1d(n_steps, dt, order):
    """Create one-dimensional finite difference matrix.

    For example, the finite difference matrix A for the second derivative
    with respect to time is defined by:

    .. math::

        \\ddot(x) = A x

    Parameters
    ----------
    n_steps : int
        Number of steps in the trajectory

    dt : float
        Time in seconds between successive steps

    order : int
        Order of the derivative, must be either 1 (velocity),
        2 (acceleration) or 3 (jerk)

    Returns
    -------
    A : array, shape (n_steps + 2, n_steps)
        Finite difference matrix for second derivative with respect to time
    """
    if order not in [1, 2, 3]:
        raise ValueError("'order' (%d) must be either 1, 2 or 3" % order)

    super_diagonal = (np.arange(n_steps), np.arange(n_steps))
    sub_diagonal = (np.arange(2, n_steps + 2), np.arange(n_steps))

    A = np.zeros((n_steps + 2, n_steps), dtype=np.float)
    if order == 1:
        A[super_diagonal] = np.ones(n_steps)
        A[sub_diagonal] = -np.ones(n_steps)
    elif order == 2:
        A[super_diagonal] = np.ones(n_steps)
        A[sub_diagonal] = np.ones(n_steps)
        main_diagonal = (np.arange(1, n_steps + 1), np.arange(n_steps))
        A[main_diagonal] = -2 * np.ones(n_steps)
    elif order == 3:
        A[super_diagonal] = np.ones(n_steps)
        A[sub_diagonal] = -1 * np.ones(n_steps)
        super_super_diagonal = (np.arange(1, n_steps), np.arange(n_steps - 1))
        A[super_super_diagonal] = -0.5 * np.ones(n_steps - 1)
        sub_sub_diagonal = (np.arange(n_steps - 1), np.arange(1, n_steps))
        A[sub_sub_diagonal] = 0.5 * np.ones(n_steps - 1)
    return A / (dt ** order)
