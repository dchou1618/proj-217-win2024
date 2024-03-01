import numpy as np

'''
(1) Implement tabular model method by building up the transition matrix directly using the 
sampled transitions.
'''


'''
(2) Implement sampling from the transitions of Geo/Geo/1/queue, specified in section D page 17 of the paper.
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
		step_addon = np.random.choice(a=3, p=[qf*(1-qa), qa*qf+(1-qa)*(1-qf),
											  qa*(1-qf)])-1
		return curr_state + step_addon

def sample_nsteps(n, initial_state, qa, qf):
	curr_state = initial_state
	state_seq = [curr_state]
	for step in range(n):
		curr_state = gg1_step(curr_state, qa, qf)
		state_seq.append(curr_state)
	return state_seq

if __name__ == "__main__":
	state_seq = sample_nsteps(n=200, initial_state=0, qa=0.8, qf=0.9)
	print(state_seq)

