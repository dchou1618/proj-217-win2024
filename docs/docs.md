---
layout: page
title: Documentation
---
## Overview

This documentation describes the structure, assumptions, and usage of the queueing and stationary distribution estimation components in this repository.

The project focuses on **simulation-based analysis of stochastic queueing systems**, with emphasis on long-run (stationary) behavior.

---

## Getting Started

### Prerequisites

- Python 3.8+
- NumPy
- Matplotlib (optional, for visualization)

Install dependencies:

```bash
pip install numpy matplotlib
```

## Overview

This project studies **stationary distribution estimation for Markov chains and queueing systems** using both classical tabular methods and a **Variational Power Method (VPM)** based on neural density ratios.

The core setting is a **Geo/Geo/1 queue**, where transitions are sampled and the stationary distribution is estimated from data.

---

## Problem Setting

Let \( X_t \) be a Markov chain with unknown transition kernel \( P \).
We aim to estimate the stationary distribution \( \pi \), satisfying:

\[
\pi = \pi P
\]

when:
- The state space may be large
- Transitions are observed via samples
- The transition matrix is not explicitly known

---

## Queueing Model (`queueing.py`)

The queueing system implements a **discrete-time Geo/Geo/1 queue**.

### Transition Dynamics

Let:
- \( q_a \): arrival probability
- \( q_f \): service completion probability

The transition probabilities are:

- Decrease by 1 with probability \( q_f (1 - q_a) \)
- Stay the same with probability \( q_a q_f + (1 - q_a)(1 - q_f) \)
- Increase by 1 with probability \( q_a (1 - q_f) \)

Implemented in:

```python
def gg1_step(curr_state, qa, qf)
```