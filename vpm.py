import torch
from torch import nn
import numpy as np
from collections import defaultdict
import queueing

'''
(1) Implement variational power method based on paper
'''

class Tau(nn.Module):
	def __init__(self, total_states):
		super().__init__()
		# 4 hidden layers
		self.model = nn.Sequential(
			nn.Linear(total_states, 32),
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



# based on paper and https://github.com/bmazoure/batch_stationary_distribution

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
		# print(f"Power step {t+1}")
		alpha_t = 1/np.sqrt(t+1)
		for _ in range(M):
			# fit transition data
			x = []
			x_star = []
			transition_batch = np.random.choice(len(D), size=B, replace=False)
			sampled_transitions = D[transition_batch]
			x = sampled_transitions[:,0]
			x_star = sampled_transitions[:, 1]
			# One-hot encoding
			x = torch.FloatTensor(np.array([[1 if node_val == i else 0 for i in range(max_state+1)] for node_val in x]))
			x_star = torch.FloatTensor(np.array([[1 if node_val == i else 0 for i in range(max_state+1)] for node_val in x_star]))
			grad_theta_x, grad_theta_xstar = defaultdict(list),defaultdict(list)
			tau_x = tau(x)
			tau_xstar = tau(x_star)
			tau_t_x = tau_t(x)
			tau_t_xstar = tau_t(x_star)
			# Populate the gradients
			for i in range(len(x)):
				optim_theta.zero_grad()
				tau_x.backward([torch.FloatTensor([[1] if i==j else [0] for j in range(len(tau_x))]) ], retain_graph=True)
				for param in tau.named_parameters():
					grad_theta_x[param[0]].append(param[1].grad.clone())
				optim_theta.zero_grad()
				tau_xstar.backward([torch.FloatTensor([[1] if i==j else [0] for j in range(len(tau_xstar))])], retain_graph=True)
				for param in tau.named_parameters():
					grad_theta_xstar[param[0]].append(param[1].grad.clone())

			optim_theta.zero_grad()
			optim_v.zero_grad()
			
			for param in tau.named_parameters():
				# first dimension is B
				grad_theta_tau_x_mat = torch.stack(grad_theta_xstar[param[0]])
				grad_theta_tau_xstar_mat = torch.stack(grad_theta_x[param[0]])
				
				if len(grad_theta_tau_x_mat.shape) == 3:
					tiled_tau_xstar =  tau_xstar.repeat(grad_theta_tau_xstar_mat.shape[1],1,grad_theta_tau_xstar_mat.shape[2]).permute(1,0,2)
					tiled_tau_t_x =  tau_t_x.repeat(grad_theta_tau_xstar_mat.shape[1],1,grad_theta_tau_xstar_mat.shape[2]).permute(1,0,2)
					tiled_tau_t_xstar =  tau_t_xstar.repeat(grad_theta_tau_xstar_mat.shape[1],1,grad_theta_tau_xstar_mat.shape[2]).permute(1,0,2)
				else: 
					tiled_tau_xstar =  tau_xstar.repeat(1,grad_theta_tau_xstar_mat.shape[1])
					tiled_tau_t_x =  tau_t_x.repeat(1,grad_theta_tau_xstar_mat.shape[1])
					tiled_tau_t_xstar =  tau_t_xstar.repeat(1,grad_theta_tau_xstar_mat.shape[1])

				# tiled_tau_xstar = tau_xstar.tile((grad_theta_tau_xstar_mat.shape[0], ))
				# tiled_tau_t_xstar = tau_t_xstar.tile((grad_theta_tau_xstar_mat.shape[0],))
				# tiled_tau_t_x = tau_t_x.tile((grad_theta_tau_xstar_mat.shape[0],))
				# print(tiled_tau_xstar.shape, grad_theta_tau_xstar_mat.shape)
				grad_J_theta = (tiled_tau_xstar * grad_theta_tau_xstar_mat).mean(0) 
				grad_J_theta -= (1 - alpha_t) * (tiled_tau_t_xstar * grad_theta_tau_xstar_mat).mean(0) 
				grad_J_theta -= alpha_t * (tiled_tau_t_x * grad_theta_tau_xstar_mat).mean(0)
				grad_J_theta += v*grad_theta_tau_x_mat.mean(0)
				param[1].grad = grad_J_theta
			
			grad_v = ((tau_x).mean(dim=0)-1-_lambda*v)
			v.grad = -grad_v
			
			optim_theta.step()
			optim_v.step()
			

		# update and fix reference network
		tau_t.load_state_dict(tau.state_dict())
	return tau

def get_D(n_steps, qa, qf):
	transitions, max_state = queueing.sample_nsteps(n=n_steps, qa=qa,qf=qf)
	return transitions, max_state

def get_density(tau, max_state):
	density_ratios = []
	for idx in range(max_state+1):
		density_ratios.append(tau(torch.FloatTensor([[1 if i == idx else 0 for i in range(max_state+1)]])).detach().cpu().item())
	density_ratios = np.array(density_ratios)
	return density_ratios/np.sum(density_ratios)



