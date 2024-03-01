import numpy as np

'''
(1) Implement tabular model method by building up the transition matrix directly using the 
sampled transitions.
'''

def build_tabular(state_seq, max_state):
	T = np.zeros((max_state+1, max_state+1), dtype=float)
	for i in range(1, len(state_seq)):
		T[state_seq[i-1]][state_seq[i]] += 1
	row_sums = T.sum(axis=1)
	row_sums = np.reshape(row_sums, (row_sums.shape[0], 1) )
	return T/row_sums

'''
(2) Implement power method for the tabular method to estimate stationary distribution.
'''

def fronebius_norm(mat_vec_product):
	return np.sqrt(np.sum([mat_vec_product[i]**2 for i in range(len(mat_vec_product)) for j in range(len(mat_vec_product[i])) ] ))

def estimate_tabular_stationary(transition_mat, iters):
	x = np.random.rand(transition_mat.shape[1],1)
	
	print(transition_mat)
	for _ in range(iters):
		prod = np.matmul(transition_mat, x)
		x = prod/fronebius_norm(prod)
	return x/x.sum()
	

'''
(3) Implement sampling from the transitions of Geo/Geo/1/queue, specified in section D page 17 of the paper.
Serfozo 2009 for details on the closed form stationary distribution of discrete M/M/1 queue.
'''

def gg1_step(curr_state, qa, qf):
	if (curr_state == 0):
		if (np.random.uniform(low=0.0, high=1.0, size=1) < qa):
			return curr_state+1
		else:
			return curr_state
	else:
		# 0,1,2 -> -1, 0, 1 after subtracting 1
		step_addon = np.random.choice(a=3, p=[qf*(1-qa), qa*qf+(1-qa)*(1-qf), qa*(1-qf)])-1
		return curr_state + step_addon


def sample_nsteps(n, initial_state, qa, qf, transition_func=gg1_step):
	curr_state = initial_state
	state_seq = [curr_state]
	max_state = initial_state
	for _ in range(n):
		curr_state = transition_func(curr_state, qa, qf)
		state_seq.append(curr_state)
		if curr_state > max_state:
			max_state = curr_state
	return state_seq, max_state

'''
(4) Ground truth stationary distribution
'''

def ground_truth_queueing_stationary(qa, qf, max_state):
	rho = qa*(1-qf)/(qf*(1-qa))
	return [(1-rho)*(rho**i) for i in range(0, max_state+1)]
	
if __name__ == "__main__":
	qa, qf = 0.8, 0.9
	state_seq, max_state = sample_nsteps(n=200, initial_state=0, qa=qa, qf=qf)
	# print(state_seq)
	tabular_model = build_tabular(state_seq, max_state)
	# tabular_model = np.array([[0.5, 0.5], [0.2, 0.8]])
	# print(tabular_model)
	print(estimate_tabular_stationary(tabular_model, iters=500))
	print(ground_truth_queueing_stationary(qa, qf, max_state))

