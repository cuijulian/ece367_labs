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
fig4, mesh3 = plt.subplots(1, 1)
fig5, mesh4 = plt.subplots(1, 1)
fig6, mesh5 = plt.subplots(1, 1)
fig7, mesh6 = plt.subplots(1, 1)
fig8, mesh7 = plt.subplots(1, 1)
fig9, mesh8 = plt.subplots(1, 1)
fig10, mesh9 = plt.subplots(1, 1)

mesh1 = fig2.gca(projection='3d')
mesh2 = fig3.gca(projection='3d')
mesh3 = fig4.gca(projection='3d')
mesh4 = fig5.gca(projection='3d')
mesh5 = fig6.gca(projection='3d')
mesh6 = fig7.gca(projection='3d')
mesh7 = fig8.gca(projection='3d')
mesh8 = fig9.gca(projection='3d')
mesh9 = fig10.gca(projection='3d')

# Quadratic approximations
QUAD_APPROX1 = 2 * X + 3
QUAD_APPROX2 = X ** 2 + 2 * X - 4
QUAD_APPROX3 = -1.892 * X ** 2 - 2.985 * X + 2.649

# Part c)
QUAD_APPROX4 = 2 * X + 5.6
QUAD_APPROX5 = X ** 2 - 3.4 * X + 0.89
QUAD_APPROX6 = -0.826 * X ** 2 + 0.034 * X + 7.295

QUAD_APPROX7 = 2 * X + 3
QUAD_APPROX8 = X ** 2 + 6 * X + 4.75
QUAD_APPROX9 = 1.795 * X ** 2 - 3.847 * X - 5.991

mesh1.plot_surface(X, Y, QUAD_APPROX1)
mesh1.set_title('Quadratic Approximation 1')
mesh1.set_xlabel('x')
mesh1.set_ylabel('y')

mesh2.plot_surface(X, Y, QUAD_APPROX2)
mesh2.set_title('Quadratic Approximation 2')
mesh2.set_xlabel('x')
mesh2.set_ylabel('y')

mesh3.plot_surface(X, Y, QUAD_APPROX3)
mesh3.set_title('Quadratic Approximation 3')
mesh3.set_xlabel('x')
mesh3.set_ylabel('y')

mesh4.plot_surface(X, Y, QUAD_APPROX4)
mesh4.set_title('Quadratic Approximation 4')
mesh4.set_xlabel('x')
mesh4.set_ylabel('y')

mesh5.plot_surface(X, Y, QUAD_APPROX5)
mesh5.set_title('Quadratic Approximation 5')
mesh5.set_xlabel('x')
mesh5.set_ylabel('y')

mesh6.plot_surface(X, Y, QUAD_APPROX6)
mesh6.set_title('Quadratic Approximation 6')
mesh6.set_xlabel('x')
mesh6.set_ylabel('y')

mesh7.plot_surface(X, Y, QUAD_APPROX7)
mesh7.set_title('Quadratic Approximation 7')
mesh7.set_xlabel('x')
mesh7.set_ylabel('y')

mesh8.plot_surface(X, Y, QUAD_APPROX8)
mesh8.set_title('Quadratic Approximation 8')
mesh8.set_xlabel('x')
mesh8.set_ylabel('y')

mesh9.plot_surface(X, Y, QUAD_APPROX9)
mesh9.set_title('Quadratic Approximation 9')
mesh9.set_xlabel('x')
mesh9.set_ylabel('y')

plt.show()
