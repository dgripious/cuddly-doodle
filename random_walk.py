# 1D Random Walk Simulation
# Beginner-friendly script intended for a Google Colab notebook.
# At each time step, a fair coin is flipped: heads -> step right (+1),
# tails -> step left (-1). The trajectory and squared displacement
# are then plotted.

import numpy as np
import matplotlib.pyplot as plt

# --- DEFINE SIMULATION PARAMETERS ---
# N_steps controls how long the walk runs. More steps = longer walk.
N_steps = 1000

# --- DEFINE DATA STRUCTURE ---
# Pre-allocate a vector of length N_steps + 1 so we have room for the
# initial position (index 0) plus every subsequent step (1..N_steps).
# Filling with zeros also implicitly sets the starting position x[0] = 0.
x = np.zeros(N_steps + 1)
x[0] = 0  # explicit starting position for clarity

# --- UPDATE RULE ---
# Walk forward one step at a time. At each step, draw a uniform random
# number in [0, 1). If it exceeds 0.5 we move right, otherwise left.
# This is equivalent to flipping a fair coin.
for i in range(1, N_steps + 1):
    rand = np.random.random()  # random number between 0 and 1
    if rand > 0.5:
        x[i] = x[i - 1] + 1    # heads: step right
    else:
        x[i] = x[i - 1] - 1    # tails: step left

# --- DEFINE TIME VECTOR ---
# Integer time axis 0, 1, 2, ..., N_steps (same length as x).
time = np.arange(N_steps + 1)

# --- PLOT 1: trajectory ---
# Shows the walker's position as a function of time. Useful for
# visualizing how the path meanders away from the origin.
plt.figure()
plt.plot(time, x)
plt.xlabel("time step")
plt.ylabel("position x")
plt.title("Random walk trajectory")
plt.show()

# --- PLOT 2: squared displacement ---
# x^2 vs time. For a simple random walk, the *average* squared
# displacement grows linearly with time, though a single realization
# will look noisy.
plt.figure()
plt.plot(time, x ** 2)
plt.xlabel("time step")
plt.ylabel("x^2")
plt.title("Squared displacement")
plt.show()
