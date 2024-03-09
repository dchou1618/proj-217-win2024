import numpy as np
import scipy.stats
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import vpm
import estimate_stationary_dist as esd

'''
(1) Implement tabular model method by building up the transition matrix directly using the 
sampled transitions.
'''

def build_tabular(state_seq, max_state):
	T = np.zeros((max_state+1, max_state+1), dtype=float)
	for (curr_state, next_state) in state_seq:
		T[curr_state][next_state] += 1
	
	row_sums = T.sum(axis=1)
	row_sums = np.reshape(row_sums, (row_sums.shape[0], 1) )

	return T/row_sums

'''
(2) Estimate stationary distribution from matrix of sampled transitions
'''

def estimate_tabular_stationary(transition_mat, iters):
	stationary_estimate = np.zeros((1,transition_mat.shape[0]))
	for _ in range(iters):
		# many transitions and we obtain the last state visited
		state = np.random.choice(a=transition_mat.shape[0])
		for _ in range(iters):
			# print(state, transition_mat[state,:])
			# implementing a lazy random walk
			if np.isnan(transition_mat[state,:]).any():
				state = np.random.choice(a=transition_mat.shape[0])
			else:
				state = np.random.choice(a=transition_mat.shape[0], 
									 p=transition_mat[state,:])
		stationary_estimate[0, state] += 1
	
	return (stationary_estimate/stationary_estimate.sum())[0]

'''
(3) Implement sampling from the transitions of Geo/Geo/1/queue, specified in section D page 17 of the paper.
Serfozo 2009 for details on the closed form stationary distribution of discrete M/M/1 queue.
'''

def gg1_step(curr_state, qa, qf):
	if (curr_state == 0):
		if (np.random.uniform(low=0.0, high=1.0, size=1) < qa*(1-qf)):
			return curr_state+1
		else:
			return curr_state
	else:
		# 0,1,2 -> -1, 0, 1 after subtracting 1
		step_addon = np.random.choice(a=3, p=[qf*(1-qa), qa*qf+(1-qa)*(1-qf), qa*(1-qf)])-1
		return curr_state + step_addon


def sample_nsteps(n, qa, qf, transition_func=gg1_step):
	if (n < 1):
		return [], None
	rho = (qa*(1-qf))/(qf*(1-qa))
	B = int(np.ceil(40*rho))
	state_seq = []
	max_state = -1
	for i in range(n):
		curr_state = np.random.choice(a=B)
		next_state = transition_func(curr_state, qa, qf)
		state_seq.append((curr_state, next_state) )

		max_state = max(max(next_state, curr_state), max_state)
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
	_lambda = 1
	M = 20
	alpha_theta = 0.01
	alpha_v = 0.01
	beta1 = 0.5
	T = 5
	rho = (qa*(1-qf))/(qf*(1-qa))
	B = int(np.ceil(40*rho))
	log_kl_dict = {"method":[], "Samples": [], 
                  "Log KL": []}
	for transition_samples in range(min_val, max_val+1, increment):
		print(f"Generating runs for transitions samples = {transition_samples}")
		num_runs = 0
		while num_runs < runs:
			state_seq, max_state = sample_nsteps(n=transition_samples, qa=qa, qf=qf)
			D = np.array(state_seq)
			# tau = vpm.iterative_vpm(D, alpha_theta, alpha_v, T, M, B, _lambda, max_state, beta1)
			# estimated_vpm = vpm.get_density(tau, max_state)
			estimated_vpm = esd.estimate_stationary(D, max_state)

			tabular_model = build_tabular(state_seq, max_state)
			estimated = estimate_tabular_stationary(tabular_model, iters=200)
			# true distribution only up to a maximum state.
			true = ground_truth_queueing_stationary(qa, qf, max_state)
			log_kl = np.log(scipy.stats.entropy(estimated, true))
			log_kl_vpm = np.log(scipy.stats.entropy(estimated_vpm, true))
			if (log_kl != float("inf") and log_kl_vpm != float("inf")):
				num_runs+=1
				print(f"On run {num_runs}")
				#print(estimated, true)
				log_kl_dict["method"].append("Baseline Model")
				log_kl_dict["Samples"].append(transition_samples)
				log_kl_dict["Log KL"].append(log_kl)
				log_kl_dict["method"].append("Power")
				log_kl_dict["Samples"].append(transition_samples)
				log_kl_dict["Log KL"].append(log_kl_vpm)

	log_kl_pd = pd.DataFrame(log_kl_dict)
	plt.clf()
	log_kl_plot = sns.lineplot(data=log_kl_pd, x="Samples", y="Log KL", hue="method", marker="o", palette=["darkblue","firebrick"], markersize=8)
	log_kl_plot.set(ylim=(-4, 5))
	plt.grid()
	fig = log_kl_plot.get_figure()
	fig.savefig(f"{plot_name}.png") 
	
def generate_plots_over_qf(qa, min_val, max_val,  runs, plot_name):
	log_kl_dict = {"method":[], "Prob": [], "Log KL":[]}
	qfs = [np.round(min_val+(0.01)*i, 2) for i in range(int((max_val-min_val)/0.01 )+1)]
	_lambda = 1
	M = 20
	alpha_theta = 0.01
	alpha_v = 0.01
	beta1 = 0.5
	T = 5

	for qf in qfs:
		print(f"Generating runs for finish probability: {qf}")
		rho = (qa*(1-qf))/(qf*(1-qa))
		B = int(np.ceil(40*rho))
		num_runs = 0
		while num_runs < runs:
			state_seq, max_state = sample_nsteps(n=200, qa=qa, qf=qf)
			tabular_model = build_tabular(state_seq, max_state)
			if (np.isnan(tabular_model).any()):
				continue

			D = np.array(state_seq)
			tau = vpm.iterative_vpm(D, alpha_theta, alpha_v, T, M, B, _lambda, max_state, beta1)
			estimated_vpm = vpm.get_density(tau, max_state)

			estimated = estimate_tabular_stationary(tabular_model, iters=200)
			true = ground_truth_queueing_stationary(qa, qf, max_state)
			log_kl = np.log(scipy.stats.entropy(estimated, true))

			log_kl_vpm = np.log(scipy.stats.entropy(estimated_vpm, true))
			if (log_kl != float("inf") and log_kl_vpm != float("inf")):
				num_runs+=1
				print(f"On run {num_runs}")
				log_kl_dict["method"].append("Baseline Model")
				log_kl_dict["Prob"].append(qf)
				log_kl_dict["Log KL"].append(log_kl)

				log_kl_dict["method"].append("Power")
				log_kl_dict["Prob"].append(qf)
				log_kl_dict["Log KL"].append(log_kl_vpm)

	log_kl_pd = pd.DataFrame(log_kl_dict)
	plt.clf()
	log_kl_plot = sns.lineplot(data=log_kl_pd, x="Prob", y="Log KL", hue="method", marker="o", 
							palette=["darkblue", "firebrick"], markersize=8)
	log_kl_plot.set(ylim=(-4, 5))
	plt.grid()
	fig = log_kl_plot.get_figure()
	fig.savefig(f"{plot_name}.png")

if __name__ == "__main__":
	# _lambda = 1
	qa = 0.8
	qf = 0.9
	# M = 10
	# alpha_theta = 0.01
	# alpha_v = 0.01
	# beta1 = 0.5
	# T = 100
	# rho = (qa*(1-qf))/(qf*(1-qa))
	# B = int(np.ceil(40*rho))
	# D, max_state = sample_nsteps(100, qa, qf)
	# D = np.array(D)
	# tau = iterative_vpm(D, alpha_theta, alpha_v, T, M, B, _lambda, max_state, beta1)

	min_transitions, max_transitions = 100, 500
	runs = 10
	"""	
	sampled_transitions, max_state = sample_nsteps(n=2000, qa=qa, qf=qf)
	# print(sampled_transitions)
	tabular_model = build_tabular(sampled_transitions, max_state)
	# print(tabular_model)
	estimated = estimate_tabular_stationary(tabular_model, iters=200)
	# true distribution only up to a maximum state.
	true = ground_truth_queueing_stationary(qa, qf, max_state)
	print(estimated)
	print(true)
	log_kl = np.log(scipy.stats.entropy(estimated, true))

	print(log_kl)
	"""

	np.random.seed(43)
	generate_plots(qa, qf, min_transitions, max_transitions, 100, "transition_samples", runs, "figure2")
	# generate_plots_over_qf(qa, 0.82, 0.90, runs, "figure2_qf")
	
