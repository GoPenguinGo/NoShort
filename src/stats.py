import numpy as np

# @GoPenguinGo: it seems tau is always np.ndarray right?
def post_var(sigma_Y: float, V_hat: float, tau: np.ndarray) -> np.ndarray:
    """Calculate the posterior variance, correspond to eq(2)

    Args:
        sigma_Y (float): sigma_Y in eq(1), sd of Yt growth
        V_hat (float): V_hat in eq(2)
        tau (np.ndarray): (t - s) in eq(2), shape (T, )

    Returns:
        np.ndarray: shape (T, )
    """
    # TODO: @chingyulin: np.ones can be optimized.
    V = sigma_Y**2 * V_hat / (sigma_Y**2 * np.ones(len(tau)) + V_hat * tau)
    return V
