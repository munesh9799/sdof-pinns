import os
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

def exact_solution(omega, xi, t, u0=1.0, v0=0.1):
    """
    Analytical solution for: u¨ + 2*xi*omega*u˙ + omega^2*u = 0  (underdamped, xi<1)
    Uses torch ops (important: avoids mixing numpy with torch tensors).
    """
    assert xi < 1.0
    omega_d = omega * torch.sqrt(torch.tensor(1.0 - xi**2, dtype=t.dtype, device=t.device))

    # Standard underdamped form:
    # u(t) = e^{-xi*omega*t} [ u0 cos(omega_d t) + ((v0 + xi*omega*u0)/omega_d) sin(omega_d t) ]
    A = u0
    B = (v0 + xi * omega * u0) / omega_d

    return torch.exp(-(xi * omega) * t) * (A * torch.cos(omega_d * t) + B * torch.sin(omega_d * t))


class FCN(nn.Module):
    "Defines a standard fully-connected network in PyTorch"

    def __init__(self, N_INPUT, N_OUTPUT, N_HIDDEN, N_LAYERS):
        super().__init__()
        activation = nn.Tanh
        self.fcs = nn.Sequential(*[
            nn.Linear(N_INPUT, N_HIDDEN),
            activation()
        ])
        self.fch = nn.Sequential(*[
            nn.Sequential(*[
                nn.Linear(N_HIDDEN, N_HIDDEN),
                activation()
            ]) for _ in range(N_LAYERS - 1)
        ])
        self.fce = nn.Linear(N_HIDDEN, N_OUTPUT)

    def forward(self, x):
        x = self.fcs(x)
        x = self.fch(x)
        x = self.fce(x)
        return x


# -------------------------
# main (workshop-style flow)
# -------------------------
os.makedirs("./results", exist_ok=True)
torch.manual_seed(123)

# define a neural network to train
pinn = FCN(1, 1, 32, 3)

# define boundary points, for the boundary loss
t_boundary = torch.tensor(0.0).view(-1, 1).requires_grad_(True)

# define training points over the entire domain, for the physics loss
t_physics = torch.linspace(0, 1, 30).view(-1, 1).requires_grad_(True)

# -------------------------
# SDOF parameters (your case)
# -------------------------
omega = 18.97366596   # rad/s
xi = 0.05
u0 = 1.0
v0 = 0.1

# map to workshop notation (mu, k) if desired
mu = 2.0 * xi * omega
k = omega ** 2

# test grid + exact
t_test = torch.linspace(0, 1, 300).view(-1, 1)
u_exact = exact_solution(omega, xi, t_test, u0=u0, v0=v0)

optimiser = torch.optim.Adam(pinn.parameters(), lr=1e-3)

for i in range(15001):
    optimiser.zero_grad()

    # compute each term of the PINN loss function
    lambda1, lambda2 = 1e-1, 1e-4

    # -------------------------
    # boundary loss: u(0)=u0, u'(0)=v0
    # -------------------------
    u = pinn(t_boundary)
    loss1 = (torch.squeeze(u) - u0) ** 2

    dudt = torch.autograd.grad(u, t_boundary, torch.ones_like(u), create_graph=True)[0]
    loss2 = (torch.squeeze(dudt) - v0) ** 2

    # -------------------------
    # physics loss: u¨ + mu*u˙ + k*u = 0
    # -------------------------
    u = pinn(t_physics)
    dudt = torch.autograd.grad(u, t_physics, torch.ones_like(u), create_graph=True)[0]
    d2udt2 = torch.autograd.grad(dudt, t_physics, torch.ones_like(dudt), create_graph=True)[0]
    loss3 = torch.mean((d2udt2 + mu * dudt + k * u) ** 2)

    # joint loss + step
    loss = loss1 + lambda1 * loss2 + lambda2 * loss3
    loss.backward()
    optimiser.step()

    # plot as training progresses (same style as workshop)
    if i % 5000 == 0:
        u_pred = pinn(t_test).detach()
        plt.figure(figsize=(6, 2.5))
        plt.scatter(t_physics.detach()[:, 0],
                    torch.zeros_like(t_physics)[:, 0], s=20, lw=0,
                    color="tab:green", alpha=0.6)
        plt.scatter(t_boundary.detach()[:, 0],
                    torch.zeros_like(t_boundary)[:, 0], s=20, lw=0,
                    color="tab:red", alpha=0.6)
        plt.plot(t_test[:, 0], u_exact[:, 0], label="Exact solution", color="tab:grey", alpha=0.6)
        plt.plot(t_test[:, 0], u_pred[:, 0], label="PINN solution", color="tab:green")
        plt.title(f"Training step {i}")
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"./results/forward_iter_{i}.png", dpi=200)
        plt.show()
