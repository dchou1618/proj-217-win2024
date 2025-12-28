import torch
import numpy as np
import torch.nn as nn
from collections import defaultdict
import queueing

"""
Implementation by Bogdan Mazoure
Paper: https://arxiv.org/abs/2003.00722
"""

class Tau(nn.Module):
    def __init__(self,input_dim):
        super().__init__()
        self.model = nn.Sequential(
			nn.Linear(input_dim, 32),
			nn.ReLU(),
			nn.Linear(32, 32),
			nn.ReLU(),
            nn.Linear(32, 32),
			nn.ReLU()
		)
        self.output_layer = nn.Sequential(nn.Linear(32, 1),
										  nn.Softplus())
    def forward(self, x):
        x = self.model(x)
        x = self.output_layer(x)
        return x
    
def fit_tau(X_t,X_tp1, tau, tau_t, vee, opt_tau, opt_vee, epoch):
    """
    Meant to be called on a batch of states to estimate the batch gradient wrt tau and v.
    """
    alpha_t = 1. / np.sqrt(epoch+1)
    lam = 1

    tau_xt = tau(X_t).squeeze()
    tau_xtp1 = tau(X_tp1).squeeze()
    tau_t_xt = tau_t(X_t).squeeze()
    tau_t_xtp1 = tau_t(X_tp1).squeeze()
    J_tau = (
        (tau_xtp1 ** 2).mean()
        - (1 - alpha_t) * (tau_t_xtp1 * tau_xtp1).mean()
        - alpha_t * (tau_t_xt * tau_xt).mean()
        + vee * tau_xt.mean()
    )

    J_v = lam * (2 * vee * (tau_xt.mean() - 1) - vee**2)

    opt_tau.zero_grad()
    opt_vee.zero_grad()

    J_tau.backward(retain_graph=True)
    J_v.backward()

    opt_tau.step()
    opt_vee.step()

    # Do soft update on the reference network
    with torch.no_grad():
        tau_t_state = tau_t.state_dict()
        tau_state = tau.state_dict()
        for key in tau_state.keys():
            tau_t_state[key] = 0.99 * tau_t_state[key] + 0.01 * tau_state[key]
        tau_t.load_state_dict(tau_t_state)



def get_D(n_steps, qa, qf):
	transitions, max_state = queueing.sample_nsteps_non_iid(n=n_steps, initial_state=0, qa=qa,qf=qf)
	return transitions, max_state

def estimate_stationary(D, max_state):
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

    tau_t.load_state_dict(tau.state_dict())

    opt_tau = torch.optim.Adam(tau.parameters(),lr=1e-3)
    opt_vee = torch.optim.Adam([vee],lr=1e-3)
    

    for epoch in range(500):
        fit_tau(X_t,X_tp1, tau, tau_t, vee, opt_tau, opt_vee, epoch)
        density_ratios = []
        with torch.no_grad():
            for idx in range(max_state+1):
                input_vec = torch.zeros(max_state+1)
                input_vec[idx] = 1.0
                input_vec = input_vec.unsqueeze(0).to(device)
                density_ratios.append(tau(input_vec).item())
        density_ratios = np.array(density_ratios)
    return density_ratios/np.sum(density_ratios)

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
    n_steps = 500
    D, max_state = get_D(n_steps, qa, qf)
    D = np.array(D)

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
    
    
