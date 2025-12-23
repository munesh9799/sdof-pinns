import sys
from pathlib import Path

import torch
import matplotlib.pyplot as plt

# allow imports from src/
sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from analytic import exact_solution
from pinn import FCN, boundary_losses, physics_loss
from data import make_physics_points, make_test_grid
from utils import rmse, savefig, write_log


def main():
    torch.manual_seed(123)

    # domain points
    t0 = torch.tensor([[0.0]], requires_grad=True)
    t_physics = make_physics_points(n_phys=30)
    t_test = make_test_grid(n_test=300)

    # parameters (workshop defaults)
    d, w0 = 2.0, 20.0
    mu, k = 2.0 * d, w0**2

    # exact
    u_exact = exact_solution(d, w0, t_test)

    # PINN
    pinn = FCN(1, 1, 32, 3)
    optimiser = torch.optim.Adam(pinn.parameters(), lr=1e-3)

    # train
    lambda1, lambda2 = 1e-1, 1e-4
    losses = []

    for i in range(15001):
        optimiser.zero_grad()

        loss1, loss2 = boundary_losses(pinn, t0)
        loss3 = physics_loss(pinn, t_physics, mu=mu, k=k, m=1.0)

        loss = loss1 + lambda1 * loss2 + lambda2 * loss3
        loss.backward()
        optimiser.step()

        if i % 250 == 0:
            losses.append(float(loss.detach()))
            print(f"step {i:5d} | loss {loss.item():.6e}")

    # evaluate
    u_pinn = pinn(t_test).detach()
    e = rmse(u_pinn, u_exact)

    # plot solution
    plt.figure()
    plt.title("Forward PINN: u(t)")
    plt.plot(t_test[:, 0], u_exact[:, 0], label="Exact solution", color="tab:grey", alpha=0.7)
    plt.plot(t_test[:, 0], u_pinn[:, 0], label="PINN", color="tab:blue")
    plt.legend()
    plt.xlabel("t")
    plt.ylabel("u(t)")
    savefig("forward_solution.png")
    plt.show()

    # plot loss
    plt.figure()
    plt.title("Training loss (logged every 250 steps)")
    plt.plot(losses)
    plt.xlabel("log step")
    plt.ylabel("loss")
    savefig("forward_loss.png")
    plt.show()

    write_log(
        "forward_summary.txt",
        f"Forward PINN\n"
        f"delta={d}, w0={w0}, mu={mu}, k={k}\n"
        f"RMSE(u_pinn, u_exact) = {e:.6e}\n"
    )


if __name__ == "__main__":
    main()
