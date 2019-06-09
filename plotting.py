import matplotlib.pyplot as plt

def plot(u_h, V):
    """ Plot the solution projected in 3D at the dof location """
    from mpl_toolkits.mplot3d import axes3d
    fig = plt.figure()
    axes = fig.add_subplot(1,1,1, projection="3d")
    u_plot = u_h.reshape(V.mesh.grid[0].shape)
    mycmap = plt.get_cmap('viridis')
    surface = axes.plot_surface(V.mesh.grid[0],
                                V.mesh.grid[1], u_plot,cmap=mycmap)
    fig.colorbar(surface,ax=axes)
    axes.set_xlabel('$x$', fontsize=20)
    axes.set_ylabel('$y$', fontsize=20)
    axes.set_zlabel('$u_h$', fontsize=20)
    plt.savefig("u_h.png")
