import numpy as np
import scipy.stats

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
(2) Estimate stationary distribution from matrix of sampled transitions
'''

def estimate_tabular_stationary(transition_mat, iters):
	eigen_values, eigen_vectors = np.linalg.eig(transition_mat)

	index_ordering = sorted(enumerate(eigen_values), key=lambda x: x[1], reverse=True)
	index_ordering = [idx for idx, _ in index_ordering]
	eigen_vectors = eigen_vectors[:][:, index_ordering]
	# Moore penrose inverse to get the left eigenvectors of the matrix.
	eigen_vectors = np.matmul(np.linalg.inv(np.matmul(np.transpose(eigen_vectors), 
							  eigen_vectors) ), np.transpose(eigen_vectors))

	stationary_estimate = eigen_vectors[0,:]/eigen_vectors[0,:].sum()
	return stationary_estimate

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
	rho = (qa*(1-qf))/(qf*(1-qa))
	return [(1-rho)*(rho**i) for i in range(0, max_state+1)]
	
if __name__ == "__main__":
	qa, qf = 0.8, 0.9
	n_samples = 100
	log_kl_sum = 0
	runs = 10
	for i in range(runs):
		state_seq, max_state = sample_nsteps(n=n_samples, initial_state=0, qa=qa, qf=qf)
		tabular_model = build_tabular(state_seq, max_state)
		estimated = estimate_tabular_stationary(tabular_model, iters=1000)
		# true distribution only up to a maximum state.
		true = ground_truth_queueing_stationary(qa, qf, max_state)
		log_kl_sum += np.log(scipy.stats.entropy(estimated, true))
	print(f"Log-kl-divergence for {n_samples}: {log_kl_sum/runs}")


