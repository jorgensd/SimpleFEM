def main(nx, ny, f, quad_degree=4):
    from mesh import UnitSquareMesh
    from functionspace import FunctionSpace
    from assembly import (assemble_stiffness_matrix, assemble_rhs)

    mesh = UnitSquareMesh(nx, ny)
    V = FunctionSpace(mesh)

    A = assemble_stiffness_matrix(V, quad_degree=quad_degree)

    L = assemble_rhs(V, f, quad_degree=quad_degree)

    def bc_apply(A, b):
        # Apply a zero-dirichlet condition
        b_dofs = V.on_boundary()
        for i in b_dofs:
            A[i, :] = 0
            A[i, i] = 1
            b[i] = 0
        return A

    bc_apply(A, L)
    from numpy.linalg import solve
    u_h = solve(A, L)

    return u_h, V


if __name__ == "__main__":
    f = "4*(-y**2+y)*sin(pi*x)"
    # f = "8*pi**2*sin(2*pi*x)*sin(2*pi*y)"
    # f = "x**2+cos(2*pi*y)"

    def f_internal(x, y):
        from sympy import pi, sin, cos
        return eval(f)

    u_h, V = main(25, 10, f_internal, quad_degree=2)

    from plotting import plot_at_vertex, plot_custom, plot_contour
    plot_at_vertex(u_h, V)
    plot_custom(u_h, V, 40)
    plot_contour(u_h, V)
