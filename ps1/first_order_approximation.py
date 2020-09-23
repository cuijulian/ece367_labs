import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d

x_list = np.linspace(-2.0, 3.5, 1000)
y_list = np.linspace(-2.0, 3.5, 1000)
X, Y = np.meshgrid(x_list, y_list)

# Functions
Z1 = 2 * X + 3 * Y + 1
Z2 = X**2 + Y**2 - X * Y - 5
Z3 = (X - 5)*np.cos(Y - 5) - (Y - 5)*np.sin(X - 5)

# Create subplots for each function
fig, (f1, f2, f3) = plt.subplots(1, 3)

f1.contour(X, Y, Z1)
# Gradient of f1 at (1, 0) is [2, 3]
f1.quiver(1, 0, 2, 3)
f1.set_title('Contour Plot 1')
f1.set_xlabel('x')
f1.set_ylabel('y')

f2.contour(X, Y, Z2)
# Gradient of f2 at (1, 0) is [2, -2]
f2.quiver(1, 0, 2, -2)
f2.set_title('Contour Plot 2')
f2.set_xlabel('x')
f2.set_ylabel('y')

f3.contour(X, Y, Z3)
# Gradient of f3 at (1, 0) is about [-2.98, 3.08]
f3.quiver(1, 0, -2.98, 3.08)
f3.set_title('Contour Plot 3')
f3.set_xlabel('x')
f3.set_ylabel('y')

# Create plots for each function
fig2, mesh1 = plt.subplots(1, 1)
fig3, mesh2 = plt.subplots(1, 1)
fig4, mesh2 = plt.subplots(1, 1)

mesh1 = fig2.gca(projection='3d')
mesh2 = fig3.gca(projection='3d')
mesh3 = fig4.gca(projection='3d')

# Find tangent surfaces with linear approximations
TAN1 = 3 + 2 * (X - 1) + 3 * Y
TAN2 = -4 + 2 * (X - 1) - 2 * Y
TAN3 = 2.649 - 2.98 * (X - 1) + 3.08 * Y

mesh1.plot_surface(X, Y, Z1)
mesh1.plot_surface(X, Y, TAN1)
mesh1.set_title('Mesh Plot 1')
mesh1.set_xlabel('x')
mesh1.set_ylabel('y')

mesh2.plot_surface(X, Y, Z2)
mesh2.plot_surface(X, Y, TAN2)
mesh2.set_title('Mesh Plot 2')
mesh2.set_xlabel('x')
mesh2.set_ylabel('y')

mesh3.plot_surface(X, Y, Z3)
mesh3.plot_surface(X, Y, TAN3)
mesh3.set_title('Mesh Plot 3')
mesh3.set_xlabel('x')
mesh3.set_ylabel('y')

plt.show()
