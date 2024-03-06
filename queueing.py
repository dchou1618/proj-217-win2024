import numpy as np
import scipy.stats
import seaborn as sns
import pandas as pd

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

'''
(5) Plot generation
'''

def generate_plots(qa, qf, min_val, max_val, increment, var_name, runs, plot_name):
	log_kl_dict = {"method":[], var_name: [], 
                  "log-kl-div": []}
	for transition_samples in range(min_val, max_val+1, increment):
		num_runs = 0
		while num_runs < runs:
			state_seq, max_state = sample_nsteps(n=transition_samples, initial_state=0, qa=qa, qf=qf)
			tabular_model = build_tabular(state_seq, max_state)
			
			estimated = estimate_tabular_stationary(tabular_model, iters=1000)
			# true distribution only up to a maximum state.
			true = ground_truth_queueing_stationary(qa, qf, max_state)
			log_kl = np.log(scipy.stats.entropy(estimated, true))
			if (log_kl != float("inf")):
				num_runs+=1

				log_kl_dict["method"].append("model-based")
				log_kl_dict["transition_samples"].append(transition_samples)
				log_kl_dict["log-kl-div"].append(log_kl)
	log_kl_pd = pd.DataFrame(log_kl_dict)
	log_kl_plot = sns.lineplot(data=log_kl_pd, x="transition_samples", y="log-kl-div", hue="method", marker="o")
	log_kl_plot.set(ylim=(-5, 0))
	fig = log_kl_plot.get_figure()
	fig.savefig(f"{plot_name}.png") 
	
def generate_plots_over_qf(qa, min_val, max_val,  runs, plot_name):
	log_kl_dict = {"method":[], "qf": [], 
	"log-kl-div":[]}
	qfs = [np.round(min_val+(0.01)*i, 2) for i in range(int((max_val-min_val)/0.01 )+2)]

	for qf in qfs:
		num_runs = 0
		while num_runs < runs:
			state_seq, max_state = sample_nsteps(n=200, initial_state=0, qa=qa, qf=qf)
			tabular_model = build_tabular(state_seq, max_state)
			print(tabular_model)
			if (np.linalg.matrix_rank(tabular_model) != tabular_model.shape[0]):
				continue
			estimated = estimate_tabular_stationary(tabular_model, iters=1000)
			true = ground_truth_queueing_stationary(qa, qf, max_state)
			log_kl = np.log(scipy.stats.entropy(estimated, true))
			if (log_kl != float("inf")):
				num_runs+=1
				log_kl_dict["method"].append("model-based")
				log_kl_dict["qf"].append(qf)
				log_kl_dict["log-kl-div"].append(log_kl)
	log_kl_pd = pd.DataFrame(log_kl_dict)
	log_kl_plot = sns.lineplot(data=log_kl_pd, x="qf", y="log-kl-div", hue="method", marker="o")
	log_kl_plot.set(ylim=(-5, 0))
	fig = log_kl_plot.get_figure()
	fig.savefig(f"{plot_name}.png")

if __name__ == "__main__":
	qa, qf = 0.8, 0.9
	min_transitions, max_transitions = 100, 500
	runs = 10
	
	state_seq, max_state = sample_nsteps(n=200, initial_state=0, qa=qa, qf=qf)
	tabular_model = build_tabular(state_seq, max_state)
	estimated = estimate_tabular_stationary(tabular_model, iters=1000)
	# true distribution only up to a maximum state.
	true = ground_truth_queueing_stationary(qa, qf, max_state)
	print(estimated)
	print(true)
	log_kl = np.log(scipy.stats.entropy(estimated, true))

	print(log_kl)
	
	#generate_plots(qa, qf, min_transitions, max_transitions, 100, "transition_samples", runs, "figure2")
	#generate_plots_over_qf(qa, 0.82, 0.90, runs, "figure2_qf")
	
