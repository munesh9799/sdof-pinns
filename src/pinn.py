import torch
import torch.nn as nn


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


def grad(u, t):
    return torch.autograd.grad(u, t, torch.ones_like(u), create_graph=True)[0]


def physics_residual(pinn, t, mu, k, m=1.0):
    """
    Residual for:
      m u'' + mu u' + k u = 0
    """
    u = pinn(t)
    dudt = grad(u, t)
    d2udt2 = grad(dudt, t)
    r = m * d2udt2 + mu * dudt + k * u
    return r


def boundary_losses(pinn, t0):
    """
    Boundary/initial conditions (workshop style):
      u(0)=1, u'(0)=0
    """
    u0 = pinn(t0)
    dudt0 = grad(u0, t0)
    loss_u0 = (u0 - 1.0) ** 2
    loss_du0 = (dudt0 - 0.0) ** 2
    return torch.mean(loss_u0), torch.mean(loss_du0)


def physics_loss(pinn, t_physics, mu, k, m=1.0):
    r = physics_residual(pinn, t_physics, mu, k, m=m)
    return torch.mean(r ** 2)


def data_loss(pinn, t_obs, u_obs):
    u_pred = pinn(t_obs)
    return torch.mean((u_pred - u_obs) ** 2)
