import torch
import numpy as np
import torch.nn as nn
from collections import defaultdict
import queueing

import matplotlib.pyplot as plt

"""
Implementation by Bogdan Mazoure
Paper: https://arxiv.org/abs/2003.00722
"""

class Tau(nn.Module):
    def __init__(self,input_dim):
        super().__init__()
        self.model = nn.Sequential(
			nn.Linear(input_dim, 64),
			nn.ReLU(),
			nn.Linear(64, 32),
			nn.ReLU()
		)
        self.output_layer = nn.Sequential(nn.Linear(32, 1),
										  nn.Softplus())
    def forward(self, x):
        x = self.model(x)
        x = self.output_layer(x)
        return x

class TauTabular(nn.Module):
    def __init__(self, n):
        super().__init__()
        self.log_tau = nn.Parameter(torch.zeros(n))

    def forward(self, x):
        idx = x.argmax(dim=1)
        return torch.exp(self.log_tau[idx]).unsqueeze(1)


def fit_tau(X_t,X_tp1, tau, tau_t, vee, opt_tau, opt_vee, epoch):
    """
    Meant to be called on a batch of states to estimate the batch gradient wrt tau and v.
    """

    tau_xt = tau(X_t).squeeze()
    tau_xtp1 = tau(X_tp1).squeeze()
    with torch.no_grad():
        tau_t_xt = tau_t(X_t).squeeze()
    J_tau = (
        (tau_xtp1 ** 2).mean()
        - (tau_t_xt * tau_xtp1).mean()
        + 2 * vee * tau_xt.mean()
    )

    J_v = -(vee * (tau_xt.mean() - 1))

    opt_tau.zero_grad()
    opt_vee.zero_grad()

    J_tau.backward(retain_graph=True)
    J_v.backward()

    opt_tau.step()
    opt_vee.step()

    return J_tau.item(), J_v.item()


def get_D(n_steps, qa, qf):
	transitions, max_state = queueing.sample_nsteps_non_iid(n=n_steps, initial_state=0, qa=qa,qf=qf)
	return transitions, max_state

def estimate_stationary(D, max_state, power_iters=50, inner_steps=500, batch_size = 128):
    X_t = torch.from_numpy(D[:, 0]).long()
    X_tp1 = torch.from_numpy(D[:, 1]).long()


    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # One-hot encoding
    X_t = torch.nn.functional.one_hot(X_t, num_classes=max_state+1).float().to(device)
    X_tp1 = torch.nn.functional.one_hot(X_tp1, num_classes=max_state+1).float().to(device)

    

    tau = Tau(max_state+1)
    tau_t = Tau(max_state+1)
    vee = torch.rand(1, requires_grad=True ,device=device)
    if device == 'cuda':
        tau = tau.cuda()
        tau_t = tau_t.cuda()

    opt_tau = torch.optim.Adam(tau.parameters(),lr=3e-4)
    opt_vee = torch.optim.Adam([vee],lr=3e-4)
    losses_tau, losses_v = [], []
    for t in range(power_iters):
        tau_t.load_state_dict(tau.state_dict())
        for _ in range(inner_steps):
            idx = torch.randint(0, X_t.shape[0], (batch_size,), device=X_t.device)
            J_tau_item, J_v_item = fit_tau(X_t[idx],X_tp1[idx], tau, tau_t, vee, opt_tau, opt_vee, t)
            losses_tau.append(J_tau_item)
            losses_v.append(J_v_item)
            with torch.no_grad():
                vee.clamp_(min=0.0)
    plt.plot(losses_tau, label='J_tau')
    plt.plot(losses_v, label='J_v')
    plt.legend()
    plt.show()
    density_ratios = []
    with torch.no_grad():
        for idx in range(max_state+1):
            input_vec = torch.zeros(max_state+1)
            input_vec[idx] = 1.0
            input_vec = input_vec.unsqueeze(0).to(device)
            density_ratios.append(tau(input_vec).item())
    density_ratios = np.array(density_ratios)
    pi = density_ratios * np.bincount(D[:,0], minlength=max_state+1)
    pi /= pi.sum()
    return pi

if __name__ == "__main__":
    """
    Simple test with a 3x3 T. Its stationary vector is (0.25,0.5,0.25),
    so tau(0),tau(1),tau(2) / (tau(0)+tau(1)+tau(2)) should converge to this
    """
    import scipy.stats
    qa = 0.8
    qf = 0.9
    rho = (qa*(1-qf))/(qf*(1-qa))
    B = int(np.ceil(40*rho))
    n_steps = 20000
    burn_in = 5000
    D, max_state = get_D(n_steps, qa, qf)
    D = np.array(D)[burn_in:, :]
    np.random.shuffle(D)

    # Estimate stationary distribution using VPM
    estimated_dist = estimate_stationary(D, max_state)

    # Ground truth stationary distribution for Geo/Geo/1 queue (truncated at max_state)
    rho = (qa * (1 - qf)) / (qf * (1 - qa))
    true_dist = np.array([(1 - rho) * (rho ** i) for i in range(max_state + 1)])

    # Normalize just in case (true_dist should already sum to ~1)
    true_dist /= true_dist.sum()

    # Print both distributions for comparison
    print("Estimated stationary distribution (VPM):")
    print(np.round(estimated_dist, 4))
    print("\nGround truth stationary distribution:")
    print(np.round(true_dist, 4))

    # Calculate KL divergence (true || estimated)
    kl_divergence = scipy.stats.entropy(true_dist, estimated_dist)
    print(f"\nKL divergence (true || estimated): {kl_divergence:.6f}")
    
    
