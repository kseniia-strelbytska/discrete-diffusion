import torch
import matplotlib.pyplot as plt
import numpy as np

def get_gaussian_noise_schedule(t_i, sigma, max_l, device):
    """
    Generates a Gaussian noise schedule for a given timestep.

    Args:
        t_i (float): Normalized timestep in [0, 1].
        sigma (float): The standard deviation (width of the soft boundary).
        max_l (int): The maximum length of the sequence. 
        device (torch.device): The device to place the tensors on.
    """
    x = torch.arange(max_l, dtype=torch.float32, device=device) # shape (max_l,)
    
    # The mean shifts from max_l to 0 as t_i goes from 0 to 1
    mean = (1 - t_i) * max_l 
    
    dist = torch.distributions.Normal(mean, sigma)
    
    # Probability of being masked
    cdf = dist.cdf(x)
    p_mask = cdf 
    
    # Derivative of p_mask with respect to timestep t_i
    pdf = dist.log_prob(x).exp()
    p_mask_dt = max_l * pdf

    return p_mask, p_mask_dt

def plot_gaussian_noise_schedule(timesteps, sigma, max_l):
    fig, axes = plt.subplots(len(timesteps), 1, figsize=(10, 4 * len(timesteps)))
    
    for idx, t_i in enumerate(timesteps):
        p_mask, dp_mask_dt = get_gaussian_noise_schedule(t_i=t_i, sigma=sigma, max_l=max_l, device='cpu')
        ax = axes[idx]
        ax.plot(p_mask.numpy(), label='P(Masked) [1 - CDF]')
        ax.plot(dp_mask_dt.numpy(), label='Derivative w.r.t time', linestyle='--')
        ax.set_title(f'Timestep t_i = {t_i:.2f}')
        ax.legend()
        
    plt.tight_layout()
    plt.show()
    
def main():
    T = 100
    sigma = 4.0
    max_l = 258
    plot_gaussian_noise_schedule(timesteps=np.linspace(0, 1, 4), sigma=sigma, max_l=max_l)
    
if __name__ == "__main__":
    main()