import numpy as np


def make_demonstrations(n_demonstrations, n_steps, sigma=0.25, mu=0.5,
                        start=np.zeros(2), goal=np.ones(2), random_state=None):
    """Generates demonstration that can be used to test imitation learning.

    Parameters
    ----------
    n_demonstrations : int
        Number of noisy demonstrations

    n_steps : int
        Number of time steps

    sigma : float, optional (default: 0.25)
        Standard deviation of noisy component

    mu : float, optional (default: 0.5)
        Mean of noisy component

    start : array, shape (2,), optional (default: 0s)
        Initial position

    goal : array, shape (2,), optional (default: 1s)
        Final position

    random_state : int
        Seed for random number generator

    Returns
    -------
    X : array, shape (n_task_dims, n_steps, n_demonstrations)
        Noisy demonstrated trajectories

    ground_truth : array, shape (n_task_dims, n_steps)
        Original trajectory
    """
    random_state = np.random.RandomState(random_state)

    X = np.empty((2, n_steps, n_demonstrations))

    # Generate ground-truth for plotting
    ground_truth = np.empty((2, n_steps))
    T = np.linspace(-0, 1, n_steps)
    ground_truth[0] = T
    ground_truth[1] = (T / 20 + 1 / (sigma * np.sqrt(2 * np.pi)) *
                       np.exp(-0.5 * ((T - mu) / sigma) ** 2))

    # Generate trajectories
    for i in range(n_demonstrations):
        noisy_sigma = sigma * random_state.normal(1.0, 0.1)
        noisy_mu = mu * random_state.normal(1.0, 0.1)
        X[0, :, i] = T
        X[1, :, i] = T + (1 / (noisy_sigma * np.sqrt(2 * np.pi)) *
                          np.exp(-0.5 * ((T - noisy_mu) /
                                         noisy_sigma) ** 2))

    # Spatial alignment
    current_start = ground_truth[:, 0]
    current_goal = ground_truth[:, -1]
    current_amplitude = current_goal - current_start
    amplitude = goal - start
    ground_truth = ((ground_truth.T - current_start) * amplitude /
                    current_amplitude + start).T

    for demo_idx in range(n_demonstrations):
        current_start = X[:, 0, demo_idx]
        current_goal = X[:, -1, demo_idx]
        current_amplitude = current_goal - current_start
        X[:, :, demo_idx] = ((X[:, :, demo_idx].T - current_start) *
                             amplitude / current_amplitude + start).T

    return X, ground_truth

