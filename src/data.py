import torch
from analytic import exact_solution


def make_observations(delta, w0, n_obs=40, noise_std=0.04, seed=123):
    """
    Same pattern as pinn_intro_workshop.py:
      t_obs ~ Uniform(0,1)
      u_obs = exact_solution + noise
    """
    torch.manual_seed(seed)
    t_obs = torch.rand(n_obs).view(-1, 1)
    u_obs = exact_solution(delta, w0, t_obs) + noise_std * torch.randn_like(t_obs)
    return t_obs, u_obs


def make_physics_points(n_phys=30):
    t_physics = torch.linspace(0, 1, n_phys).view(-1, 1).requires_grad_(True)
    return t_physics


def make_test_grid(n_test=300):
    t_test = torch.linspace(0, 1, n_test).view(-1, 1)
    return t_test
