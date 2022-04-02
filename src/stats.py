import numpy as np

# @GoPenguinGo: it seems tau is always np.ndarray right?
def post_var(sigma_Y: float, V_hat: float, tau: np.ndarray) -> np.ndarray:
    """Calculate the posterior variance, correspond to eq(2)

    Args:
        sigma_Y (float): _description_ #TODO: @GoPenguinGo
        V_hat (float): _description_
        tau (np.ndarray): (t - s) in eq(2), shape (T, )

    Returns:
        np.ndarray: shape (T, )
    """
    V = sigma_Y**2 * V_hat / (sigma_Y**2 * np.ones(len(tau)) + V_hat * tau)
    return V
