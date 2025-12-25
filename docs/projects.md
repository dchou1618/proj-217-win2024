---
layout: page
title: Projects
---

## Core Components

This project is composed of several core Python modules focused on queueing theory and stationary distribution estimation.

### 📊 Stationary Distribution Estimation

**`estimate_stationary_dist.py`**  
Implements methods for estimating the stationary distribution of a queueing system via simulation.

- Focuses on long-run behavior of stochastic processes
- Useful for validating analytical queueing results

🔗 [View source code]({{ "/blob/HEAD/estimate_stationary_dist.py" | relative_url }})

---

### 🚦 Queueing Model

**`queueing.py`**  
Defines the core queueing system logic, including arrivals, service processes, and state transitions.

- Modular queueing abstractions
- Designed to support simulation-based analysis

🔗 [View source code]({{ "/blob/HEAD/queueing.py" | relative_url }})

---

### 📈 Visits Per Minute (VPM) Analysis

**`vpm.py`**  
Provides utilities for modeling and analyzing visit rates and throughput metrics.

- Useful for performance modeling
- Can be extended to real-world traffic or service systems

🔗 [View source code]({{ "/blob/HEAD/vpm.py" | relative_url }})

---

## Repository

You can browse the full repository here:

🔗 [GitHub Repository](https://github.com/dchou1618/proj-217-win2024)