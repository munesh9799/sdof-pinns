import sys
from pathlib import Path

import torch
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import least_squares

# allow imports from src/
sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from analytic import exact_solution
from pinn import FCN, physics_loss, data_loss
from data import make_observations, make_physics_points, make_test_grid
from utils import rmse, savefig, write_log


def analytic_fit_mu(t_obs, u_obs, w0, mu0=1.0):
    """
    Baseline: fit mu by minimizing ||u_exact(mu) - u_obs|| using SciPy.
    Here: mu = 2*delta.
    """
    t_np = t_obs.detach().cpu().numpy().reshape(-1)
    u_np = u_obs.detach().cpu().numpy().reshape(-1)

    def fun(mu_arr):
        mu = float(mu_arr[0])
        delta = mu / 2.0
        t_t = torch.tensor(t_np.reshape(-1, 1), dtype=torch.float32)
        u_pred = exact_solution(delta, w0, t_t).detach().cpu().numpy().reshape(-1)
        return (u_pred - u_np)

    res = least_squares(fun, x0=np.array([mu0], dtype=float))
    return float(res.x[0])


def main():
    torch.manual_seed(123)

    # true params (workshop defaults)
    d_true, w0 = 2.0, 20.0
    mu_true, k = 2.0 * d_true, w0**2
    print(f"True value of mu: {mu_true}")

    # data
    t_obs, u_obs = make_observations(d_true, w0, n_obs=40, noise_std=0.04, seed=123)
    t_physics = make_physics_points(n_phys=30)
    t_test = make_test_grid(n_test=300)
    u_exact = exact_solution(d_true, w0, t_test)

    # plot observations
    plt.figure()
    plt.title("Noisy observational data")
    plt.scatter(t_obs[:, 0], u_obs[:, 0])
    plt.plot(t_test[:, 0], u_exact[:, 0], label="Exact solution", color="tab:grey", alpha=0.6)
    plt.legend()
    plt.xlabel("t")
    plt.ylabel("u(t)")
    savefig("inverse_observations.png")
    plt.show()

    # PINN
    torch.manual_seed(123)
    pinn = FCN(1, 1, 32, 3)

    # learn mu (workshop style)
    mu = torch.nn.Parameter(torch.zeros(1, requires_grad=True))

    optimiser = torch.optim.Adam(list(pinn.parameters()) + [mu], lr=1e-3)

    lambda_data, lambda_phys = 1e-2, 1e-4
    mus = []
    losses = []

    for i in range(15001):
        optimiser.zero_grad()

        loss_p = physics_loss(pinn, t_physics, mu=mu, k=k, m=1.0)
        loss_d = data_loss(pinn, t_obs, u_obs)

        loss = lambda_phys * loss_p + lambda_data * loss_d
        loss.backward()
        optimiser.step()

        if i % 250 == 0:
            mus.append(float(mu.detach()))
            losses.append(float(loss.detach()))
            print(f"step {i:5d} | loss {loss.item():.6e} | mu {mu.item():.6f}")

    # evaluate
    u_pinn = pinn(t_test).detach()
    e = rmse(u_pinn, u_exact)

    # analytic baseline fit
    mu_fit = analytic_fit_mu(t_obs, u_obs, w0=w0, mu0=1.0)

    # plots
    plt.figure()
    plt.title("Inverse PINN: u(t)")
    plt.plot(t_test[:, 0], u_exact[:, 0], label="Exact solution", color="tab:grey", alpha=0.7)
    plt.plot(t_test[:, 0], u_pinn[:, 0], label="PINN", color="tab:blue")
    plt.scatter(t_obs[:, 0], u_obs[:, 0], label="Noisy data", alpha=0.7)
    plt.legend()
    plt.xlabel("t")
    plt.ylabel("u(t)")
    savefig("inverse_solution.png")
    plt.show()

    plt.figure()
    plt.title("Training loss (logged every 250 steps)")
    plt.plot(losses)
    plt.xlabel("log step")
    plt.ylabel("loss")
    savefig("inverse_loss.png")
    plt.show()

    plt.figure()
    plt.title("mu estimate")
    plt.plot(mus, label="PINN estimate")
    plt.hlines(mu_true, 0, len(mus), label="True value", color="tab:green")
    plt.hlines(mu_fit, 0, len(mus), label="Analytic fit (SciPy)", color="tab:orange")
    plt.legend()
    plt.xlabel("log step")
    plt.ylabel("mu")
    savefig("inverse_mu.png")
    plt.show()

    write_log(
        "inverse_summary.txt",
        f"Inverse PINN\n"
        f"true: delta={d_true}, w0={w0}, mu={mu_true}, k={k}\n"
        f"PINN mu={float(mu.detach()):.6f}\n"
        f"Analytic fit mu={mu_fit:.6f}\n"
        f"RMSE(u_pinn, u_exact) = {e:.6e}\n"
    )


if __name__ == "__main__":
    main()
