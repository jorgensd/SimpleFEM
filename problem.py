def main(nx, ny, f):
    from mesh import UnitSquareMesh
    from functionspace import FunctionSpace
    from assembly import (assemble_stiffness_matrix, assemble_rhs,
                          assemble_volume_function)
    import sympy as sp
    import numpy as np
    mesh = UnitSquareMesh(nx, ny)
    V = FunctionSpace(mesh)

    A = assemble_stiffness_matrix(V)

    L = assemble_rhs(V, f)

    def bc_apply(A, b):
        # Apply a zero-dirichlet condition
        b_dofs = V.on_boundary()
        for i in b_dofs:
            A[i, :] = 0
            A[i, i] = 1
            b[i] = 0
        return A

    bc_apply(A, L)

    u_h = np.linalg.solve(A, L)

    return u_h, float(assemble_volume_function(u_h, V))


if __name__ == "__main__":
    f = "4*(-y**2+y)*sin(pi*x)"
    def f_internal(x, y):
        from sympy import pi, sin
        return eval(f)
    print(main(10, 10)[1])
