import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import axes3d


def plot_at_vertex(u_h, V):
    """ Plot the solution projected in 3D at the dof location """
    fig = plt.figure()
    axes = fig.add_subplot(1, 1, 1, projection="3d")
    u_plot = u_h.reshape(V.mesh.grid[0].shape)
    mycmap = plt.get_cmap('viridis')
    surface = axes.plot_surface(V.mesh.grid[0],
                                V.mesh.grid[1], u_plot, cmap=mycmap)
    fig.colorbar(surface, ax=axes)
    axes.set_xlabel('$x$', fontsize=20)
    axes.set_ylabel('$y$', fontsize=20)
    axes.set_zlabel('$u_h$', fontsize=20)
    plt.savefig("u_h.png")


def plot_custom(u_h, V, Nx):
    """
    Plot solution on a custom grid with Nx^2 elements
    """
    from mesh import UnitSquareMesh
    plot_mesh = UnitSquareMesh(Nx, Nx)
    func = np.zeros(len(plot_mesh.vertices))
    # Loop over all nodes in custom mesh
    counter = 0
    index = 0
    for y_ in np.linspace(0, 1, Nx+1):
        for x_ in np.linspace(0, 1, Nx+1):
            # Find which element the point belongs to
            index = element(x_, y_, V, last_element=index)
            # Find min/max values of element coordinates
            a, b = V.mesh.vertices[V.mesh.cells[index][0]]
            c, d = V.mesh.vertices[V.mesh.cells[index][2]]
            # Find local coordinates
            xi_loc = 2 * (x_ - a) / (c - a) - 1
            eta_loc = 2 * (y_ - b) / (d - b) - 1
            u_h_loc = sum(u_h[V.mesh.cells[index][j]] * V.basis[j]
                          for j in range(4))
            func[counter] = u_h_loc.subs([(V.xi, xi_loc), (V.eta, eta_loc)])
            counter += 1
    Z = func.reshape(plot_mesh.grid[0].shape)

    fig = plt.figure()
    axes = fig.add_subplot(1, 1, 1, projection="3d")
    mycmap = plt.get_cmap('viridis')
    surface = axes.plot_surface(plot_mesh.grid[0],
                                plot_mesh.grid[1], Z, cmap=mycmap)
    fig.colorbar(surface, ax=axes)
    axes.set_xlabel('$x$', fontsize=20)
    axes.set_ylabel('$y$', fontsize=20)
    axes.set_zlabel('$u_h$', fontsize=20)
    plt.savefig("u_h_custom.png")


def element(x_, y_, V, last_element=0):
    """
    Find which element contains x_, y_.
    Optional argument is the index of last element for potential speedup
    """
    vert = V.mesh.vertices
    # Saves alot of time if visualization grid is finer than mesh
    for i in range(last_element, len(V.mesh.cells)):
        cell = V.mesh.cells[i]
        x_cond = vert[cell[0]][0] <= x_ and x_ <= vert[cell[1]][0]
        y_cond = vert[cell[0]][1] <= y_ and y_ <= vert[cell[-1]][1]
        if x_cond and y_cond:
            return i

    # If value not found in higher index, start from 0
    for i in range(0, last_element):
        cell = V.mesh.cells[i]
        x_cond = vert[cell[0]][0] <= x_ and x_ <= vert[cell[1]][0]
        y_cond = vert[cell[0]][1] <= y_ and y_ <= vert[cell[-1]][1]
        if x_cond and y_cond:
            return i


def plot_contour(u_h, V):
    """
    2D contour plot of solution u_h.
    """

    u_plot = u_h.reshape(V.mesh.grid[0].shape)
    fig = plt.figure()
    axes = fig.add_subplot(1, 1, 1)
    plt.contourf(V.mesh.grid[0], V.mesh.grid[1], u_plot)
    axes.plot(V.mesh.grid[0], V.mesh.grid[1], 'k-', lw=0.5, alpha=0.5)
    axes.plot(V.mesh.grid[0].T, V.mesh.grid[1].T, 'k-', lw=0.5, alpha=0.5)

    cbar = plt.colorbar()
    cbar.set_label("$u_h$", fontsize=20)
    plt.axis("equal")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.savefig("u_h_contour.png")
