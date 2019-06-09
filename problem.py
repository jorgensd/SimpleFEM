from mesh import UnitSquareMesh
from functionspace import FunctionSpace
from assembly import (assemble_stiffness_matrix, assemble_rhs,
                      assemble_volume_function)
import sympy as sp
import numpy as np
from IPython import embed

def main(nx, ny):
    mesh = UnitSquareMesh(nx,ny)
    V = FunctionSpace(mesh)

    A = assemble_stiffness_matrix(V)
    x, y = sp.symbols("x y")
    f = lambda x,y: 4*(-y**2+y)*sp.sin(sp.pi*x)
    L = assemble_rhs(V, f)

    def bc_apply(A, b):
        # Apply a zero-dirichlet condition
        b_dofs = V.on_boundary()
        for i in b_dofs:
            A[i,:i] = 0
            A[i,i+1:] =0
            A[i,i] = 1
            b[i] = 0
        return A

    bc_apply(A,L)

    u_h = np.linalg.solve(A, L)

    print(assemble_volume_function(u_h, V))

main(10,2)
