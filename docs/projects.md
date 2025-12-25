---
layout: page
title: Projects
---

## Stationary Distribution Estimation for Queueing Systems

This project studies **stationary distribution estimation for Markov chains**, with a focus on **discrete-time queueing systems**, using both classical tabular methods and modern **variational learning-based approaches**.

The implementation explores how stationary distributions can be recovered from sampled transitions without explicitly knowing the transition matrix.

---

### Queueing System Simulation

📄 **`queueing.py`**  
🔗 [View source](https://github.com/dchou1618/proj-217-win2024/blob/master/queueing.py)

Implements a **Geo/Geo/1 queue** as a discrete-time Markov chain.

Key components:
- Probabilistic arrivals and service completions
- State transitions governed by queue length dynamics
- IID and non-IID (trajectory-based) transition sampling
- Tabular stationary distribution estimation as a baseline

This file provides the **data-generating process** used throughout the project.

---

### Variational Power Method (VPM)

📄 **`vpm.py`**  
🔗 [View source](https://github.com/dchou1618/proj-217-win2024/blob/master/vpm.py)

Implements a **Variational Power Method** to estimate the stationary distribution via **density ratio learning**.

Highlights:
- Neural network parameterization of the density ratio \( \tau(s) \propto \pi(s) / \mu(s) \)
- Variational objective derived from the stationary distribution fixed-point equation
- Reference network stabilization and power-iteration–style updates
- Scales better than tabular methods for large state spaces

This file contains the **primary learning-based contribution** of the project.

---

### Reference & Validation Implementation

📄 **`estimate_stationary_dist.py`**  
🔗 [View source](https://github.com/dchou1618/proj-217-win2024/blob/master/estimate_stationary_dist.py)

A reference implementation adapted from:

> *Batch Stationary Distribution Estimation*  
> Mazoure et al., 2020

Purpose:
- Faithful reproduction of the paper’s algorithm
- Validation against the optimized `vpm.py` implementation
- Clear mapping between theory and code

Used primarily for **sanity checks, debugging, and theoretical comparison**.

---

### Experimental Evaluation

Experiments compare:
- Tabular stationary estimation
- Variational Power Method estimates

Across:
- Different queue parameters
- Sample sizes
- Evaluation metrics such as KL divergence

Results demonstrate improved **sample efficiency** of VPM in larger or sparsely observed state spaces.

---

### Technologies & Methods

- Markov chains
- Queueing theory
- Variational optimization
- Neural density ratio estimation
- PyTorch
- Monte Carlo simulation

---

### Future Directions

- Extension to multi-server queues
- Continuous-state approximations
- Off-policy stationary estimation
- Integration with reinforcement learning pipelines