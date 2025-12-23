import torch


def exact_solution(delta, w0, t):
    """
    Analytical underdamped solution with ICs:
      u(0) = 1
      u'(0) = 0

    Uses the same parameterization as pinn_intro_workshop.py:
      u'' + 2*delta*u' + w0^2*u = 0
    """
    assert delta < w0, "This exact solution assumes the underdamped regime (delta < w0)."

    w = torch.sqrt(torch.tensor(w0**2 - delta**2, dtype=t.dtype, device=t.device))
    phi = torch.atan(-delta / w)
    A = 1.0 / (2.0 * torch.cos(phi))

    u = torch.exp(-delta * t) * (2.0 * A * torch.cos(phi + w * t))
    return u


def parameters_from_mu_k(mu, k, m=1.0):
    """
    Convert physical parameters (m, mu, k) to (delta, w0) used by the workshop form:

      m u'' + mu u' + k u = 0
      => u'' + (mu/m) u' + (k/m) u = 0
      => u'' + 2*delta u' + w0^2 u = 0

    delta = (mu/m)/2
    w0 = sqrt(k/m)
    """
    delta = (mu / m) / 2.0
    w0 = (k / m) ** 0.5
    return delta, w0
