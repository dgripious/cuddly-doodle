# 1D Random Walk Simulation — ensemble of independent walkers
# Beginner-friendly script intended for a Google Colab notebook.
# At each time step every walker flips its own fair coin:
# heads -> step right (+1), tails -> step left (-1).
# We then plot a subset of trajectories and the mean squared
# displacement averaged across the whole ensemble.

import numpy as np
import matplotlib.pyplot as plt

# --- DEFINE SIMULATION PARAMETERS ---
# N_steps controls how long each walk runs.
# N_walkers is how many independent walkers we simulate in parallel.
# A large ensemble lets us average out randomness and see the
# underlying diffusive behavior.
N_steps = 1000
N_walkers = 1000

# --- DEFINE DATA STRUCTURE ---
# 2D array: each row is one walker, each column is one point in time.
# Shape (N_walkers, N_steps + 1), filled with zeros so every walker
# starts at the origin (x = 0 at time 0).
x = np.zeros((N_walkers, N_steps + 1))

# --- UPDATE RULE ---
# March forward one time step at a time. At each step draw one random
# number per walker; numbers > 0.5 become +1 (right), the rest -1 (left).
# Vectorizing across walkers keeps the script to a single time loop
# and makes it fast even for large ensembles.
for i in range(1, N_steps + 1):
    rand = np.random.random(N_walkers)          # one coin per walker
    steps = np.where(rand > 0.5, 1, -1)         # map to +/- 1
    x[:, i] = x[:, i - 1] + steps               # advance every walker

# --- DEFINE TIME VECTOR ---
# Integer time axis 0, 1, 2, ..., N_steps.
time = np.arange(N_steps + 1)

# --- COMPUTE MEAN SQUARED DISPLACEMENT ---
# Average x^2 across walkers at each time. For an unbiased 1D walk
# with unit steps, theory predicts <x^2> = t (linear in time).
msd = np.mean(x ** 2, axis=0)

# --- PLOT 1: trajectories ---
# Plotting all 1000 paths would be an unreadable blob, so we show a
# subset with partial transparency. You still get a sense of the
# spreading "cone" of possible positions over time.
plt.figure()
N_show = 50
for w in range(N_show):
    plt.plot(time, x[w], alpha=0.3)
plt.xlabel("time step")
plt.ylabel("position x")
plt.title(f"Random walk: {N_show} of {N_walkers} trajectories")
plt.show()

# --- PLOT 2: mean squared displacement ---
# Compare the simulated <x^2> to the theoretical prediction y = t.
# They should agree closely for large N_walkers, confirming diffusion.
plt.figure()
plt.plot(time, msd, label="simulated <x^2>")
plt.plot(time, time, "--", label="theory: y = t")
plt.xlabel("time step")
plt.ylabel("<x^2>")
plt.title("Mean squared displacement")
plt.legend()
plt.show()
