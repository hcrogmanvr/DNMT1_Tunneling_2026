# 2D_surface_plot.py
# Generates a 3D surface plot of the 2D potential energy surface (PES)

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm

# Load 2D PES data from a file
# Assume format: x y z (one line per grid point)
data = np.loadtxt('surface_2d.dat')
X = data[:, 0]
Y = data[:, 1]
Z = data[:, 2]

# Reshape to 2D grid (adjust shape as needed)
grid_size = int(np.sqrt(len(X)))
X = X.reshape((grid_size, grid_size))
Y = Y.reshape((grid_size, grid_size))
Z = Z.reshape((grid_size, grid_size))

# Create the plot
fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')
surf = ax.plot_surface(X, Y, Z, cmap=cm.viridis, edgecolor='none')

# Labels and title
ax.set_xlabel('C–C Distance (Å)')
ax.set_ylabel('Orthogonal Coordinate (Å)')
ax.set_zlabel('Energy (kcal/mol)')
ax.set_title('2D PES Surface')

# Add color bar
fig.colorbar(surf, ax=ax, shrink=0.5, aspect=10)

# Save the figure
plt.tight_layout()
plt.savefig('2D_surface_plot.png')
plt.close()