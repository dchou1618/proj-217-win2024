import torch
from torch import nn
import numpy as np

'''
(1) Implement variational power method based on paper
'''

class Tau(nn.Module):
	def __init__(self, total_elems):
		super.__init__()
		self.flatten = nn.Flatten()
		# 4 hidden layers
		self.model = nn.Sequential(
			nn.Linear(total_elems, 128),
			nn.ReLU(),
			nn.Linear(128, 128),
			nn.ReLU(),
            nn.Linear(128, 128),
			nn.ReLU(),
            nn.Linear(128, 128),
			nn.ReLU(),
            nn.Linear(128, 128),
			nn.ReLU(),
            nn.Linear(128, 1),
			nn.Softplus()
		)
	def forward(self, x):
		return self.model(x)


def iterative_vpm(transition_data, alpha_theta, alpha_v, power_steps, M, B):
	for t in range(power_steps):
		for m in range(M):
			# fit transition data
			pass

if __name__ == "__main__":
	pass
