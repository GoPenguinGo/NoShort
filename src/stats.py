import numpy as np
from numba import jit

@jit(nopython=True)
def post_var(sigma_Y: float, V_hat: float, tau: np.ndarray) -> np.ndarray:
    """Calculate the posterior variance, correspond to eq(2)

    Args:
        sigma_Y (float): sigma_Y in eq(1), sd of Yt growth
        V_hat (float): V_hat in eq(2)
        tau (np.ndarray): (t - s) in eq(2), shape (T, )

    Returns:
        np.ndarray: shape (T, )
    """
    sigma_Y_sq = sigma_Y**2
    V = sigma_Y_sq * V_hat / (sigma_Y_sq + V_hat * tau)
    return V
