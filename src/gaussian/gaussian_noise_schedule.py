import torch
import matplotlib.pyplot as plt
import numpy as np
import time

def get_gaussian_noise_schedule(t_i, sigma, max_l, device):
    """
    Generates a Gaussian noise schedule for a given timestep.

    Args:
        t_i (float): Normalized timestep in [0, 1]. (0 = clean data, 1 = full noise)
        sigma (float): The standard deviation (width of the soft boundary).
        max_l (int): The maximum length of the sequence. 
        device (torch.device): The device to place the tensors on.
    """
    x = torch.arange(max_l, dtype=torch.float32, device=device).unsqueeze(0)
    
    margin = 2.0 * sigma
    width = max_l + 2.0 * margin
    
    # FIX 1: Reverse the sweep. 
    # At t=0 (clean), the mean is far right, CDF is ~0.
    # At t=1 (noisy), the mean is far left, CDF is ~1.
    mean = (1.0 - t_i) * width - margin
    
    dist = torch.distributions.Normal(mean, sigma)
    
    cdf = dist.cdf(x)
    p_mask = cdf 
        
    # FIX 2: The chain rule multiplier is the total sweep 'width', not 'max_l'.
    pdf = dist.log_prob(x).exp()
    p_mask_dt = pdf * width

    return p_mask, p_mask_dt

def plot_gaussian_noise_schedule(timesteps, sigma, max_l):
    fig, axes = plt.subplots(len(timesteps), 1, figsize=(10, 4 * len(timesteps)))
    
    for idx, t_i in enumerate(timesteps):
        p_mask, dp_mask_dt = get_gaussian_noise_schedule(t_i=t_i, sigma=sigma, max_l=max_l, device='cpu')
        
        ax = axes[idx]
        ax.plot(p_mask.squeeze(0).numpy(), label='P(Masked) [1 - CDF]')
        ax.plot(dp_mask_dt.squeeze(0).numpy(), label='Derivative w.r.t time', linestyle='--')
        ax.set_title(f'Timestep t_i = {t_i:.2f}')
        ax.legend()
        
    plt.tight_layout()
    plt.savefig('noise_schedule.png')  # Save as an image
    print("Plot saved to noise_schedule.png")
        
def main():
    T = 100
    sigma = 4.0
    max_l = 258
    plot_gaussian_noise_schedule(timesteps=np.linspace(0, 1, 4), sigma=sigma, max_l=max_l)
    
if __name__ == "__main__":
    main()