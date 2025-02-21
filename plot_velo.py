import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Load CSV file
data = pd.read_csv("velocity_field.csv")

# Extract data
X = data["X"].values
Y = data["Y"].values
U = data["U"].values
V = data["V"].values

# Get unique grid sizes
nx = len(np.unique(X))
ny = len(np.unique(Y))

# Reshape into grid format
X = X.reshape(ny, nx)
Y = Y.reshape(ny, nx)
U = U.reshape(ny, nx)
V = V.reshape(ny, nx)

# Plot the velocity field
plt.figure(figsize=(10, 8))
plt.quiver(X, Y, U, V, scale=5, pivot="middle", color="blue")

# Labels and title
plt.xlabel("X")
plt.ylabel("Y")
plt.title("Lid-Driven Cavity Flow (Velocity Field)")

# Save the plot as PNG
plt.savefig("velocity_field.png", dpi=300)
plt.show()

print("Velocity field plot saved as velocity_field.png")

