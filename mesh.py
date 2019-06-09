import numpy as np
class UnitSquareMesh():
    def __init__(self, nx, ny):
        """
        Return a 2D UnitSquareMesh with quadrilaterals
        nx and ny are the divisions in the x and y directions.
        """
        x = np.linspace(0, 1, nx+1)
        y = np.linspace(0, 1, ny+1)
        grid = np.meshgrid(x, y)

        vertices = np.zeros(((nx+1)*(ny+1), 2), dtype=np.float)
        cells = np.zeros((nx*ny, 4), dtype=np.int)

        vertex = 0
        for iy in range(ny+1):
            for ix in range(nx+1):
                vertices[vertex,:] = x[ix], y[iy]
                vertex += 1

        cell = 0
        for iy in range(ny):
            for ix in range(nx):
                v0 = iy*(nx + 1) + ix
                v1 = v0 + 1
                v2 = v0 + nx+1
                v3 = v1 + nx+1
                cells[cell,:] = v0, v1, v3, v2;  cell += 1

        self.cells = cells
        self.vertices = vertices
        self.grid = grid

    def on_boundary(self, x):
        """ Checking if a coordinate is on the boundary of the mesh """
        return  (np.isclose(x[0],0) or np.isclose(x[1],0)
                 or np.isclose(x[0],1) or np.isclose(x[1],1))
