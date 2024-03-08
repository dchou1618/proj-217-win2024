import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import queueing

'''
(1) Implement variational power method based on paper
'''

class Tau(nn.Module):
	def __init__(self, total_states):
		super().__init__()
		# 4 hidden layers
		self.model = nn.Sequential(
			nn.Linear(total_states, 128),
			nn.ReLU(),
			nn.Linear(128, 128),
			nn.ReLU(),
            nn.Linear(128, 128),
			nn.ReLU(),
            nn.Linear(128, 128),
			nn.ReLU(),
            nn.Linear(128, 128),
			nn.ReLU()
		)
		self.output_layer = nn.Sequential(nn.Linear(128, 1),
										  nn.Softplus())
	def forward(self, x):
		x = self.model(x)
		x = self.output_layer(x)
		return x


def compute_gradients():
	pass


def iterative_vpm(D, alpha_theta, alpha_v, power_steps, M, B, _lambda, max_state, beta1):
	tau = Tau(total_states = max_state + 1)
	# reference network
	tau_t = Tau(total_states = max_state + 1)
	# update and fix reference network
	tau_t.load_state_dict(tau.state_dict())
	v = torch.rand(1, requires_grad=True)	
	optim_theta = torch.optim.Adam(tau.parameters(), lr = alpha_theta, betas=(beta1, 0.999))
	optim_v = torch.optim.Adam([v], lr = alpha_v)

	for t in range(power_steps):
		for m in range(M):
			# fit transition data
			x = []
			x_star = []
			transition_batch = np.random.choice(len(D), size=B, replace=False)
			sampled_transitions = D[transition_batch]
			x = sampled_transitions[:,0]
			# next state
			x_star = sampled_transitions[:,1]
			data_loader = DataLoader(TensorDataset(x, x_star), batch_size=B, shuffle=True)
			for x, x_star in data_loader:
				output = tau(x)
				output.retain_grad()
				optim_theta.zero_grad()
				grad_theta, grad_v = compute_gradients(output, v, lambda_val)
			print(x_star)
			break
		# update and fix reference network
		tau_t = tau_theta.clone().detach()
	return tau

def get_D(n_steps, initial_state, qa, qf):
	state_seq, max_state = queueing.sample_nsteps(n=n_steps+1, initial_state=initial_state,qa=qa,qf=qf)
	transitions = []
	for i in range(1,len(state_seq)):
		transitions.append((state_seq[i-1], state_seq[i]))
	return transitions, max_state

if __name__ == "__main__":
	_lambda = 1
	qa = 0.8
	qf = 0.9
	M = 10
	alpha_theta = 0.0005
	alpha_v = 0.0005
	beta1 = 0.5
	T = 1000
	rho = (qa*(1-qf))/(qf*(1-qa))
	B = int(np.ceil(40*rho))
	D, max_state = get_D(100, 0, qa, qf)
	D = np.array(D)
	iterative_vpm(D, alpha_theta, alpha_v, T, M, B, _lambda, max_state, beta1)



