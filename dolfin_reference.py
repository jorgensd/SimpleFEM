def dolfin_comparasion(Nx, Ny, f):
    from dolfin import (UnitSquareMesh, FunctionSpace, TrialFunction,
                        TestFunction, assemble, inner, grad, dx,
                        SpatialCoordinate, DirichletBC, Function, solve,
                        pi, CellType, sin, cos)
    mesh = UnitSquareMesh.create(Nx, Ny, CellType.Type.quadrilateral)
    V = FunctionSpace(mesh, "CG", 1)
    u, v = TrialFunction(V), TestFunction(V)
    a = inner(grad(u), grad(v))*dx

    A_dol = assemble(a)
    x, y = SpatialCoordinate(mesh)
    f = eval(f)
    l_ = inner(f, v)*dx
    B_dol = assemble(l_)
    bc = DirichletBC(V, 0, "on_boundary")
    bc.apply(A_dol, B_dol)
    u_h = Function(V)
    solve(A_dol, u_h.vector(), B_dol)
    return u_h, assemble(u_h*dx)


if __name__ == "__main__":
    f = "4*(-y**2+y)*sin(pi*x)"
    u_h, J = dolfin_comparasion(10, 2, f)
    from dolfin import File
    File("u_h.pvd") << u_h
