import numpy as np
import sympy as sp
from scipy import special
from IPython import embed

xi = sp.Symbol("xi")
eta = sp.Symbol("eta")
basis = [(xi-1)*(eta-1)/4,
                   -(xi+1)*(eta-1)/4,
                   (xi+1)*(eta+1)/4,
                   -(xi-1)*(eta+1)/4]


def mesh(nx, ny):
    """
    Return a 2D finite element mesh on a rectangle with
    extend x and y in the x and y directions.
    nx and ny are the divisions in the x and y directions.
    Return vertices and cells (local to global vertex number mapping).
    """
    x = np.linspace(0, 1, nx+1)
    y = np.linspace(0, 1, ny+1)

    vertices = np.zeros(((nx+1)*(ny+1), 2), dtype=np.float)
    cells = np.zeros((nx*ny, 4), dtype=np.int)

    vertex = 0
    for iy in range(ny+1):
        for ix in range(nx+1):
            vertices[vertex,:] = x[ix], y[iy]
            vertex += 1


    cell = 0
    # Quadrilateral elements
    for iy in range(ny):
        for ix in range(nx):
            v0 = iy*(nx + 1) + ix
            v1 = v0 + 1
            v2 = v0 + nx+1
            v3 = v1 + nx+1
            cells[cell,:] = v0, v1, v3, v2;  cell += 1
    
    return vertices, cells

xi = sp.Symbol("xi")
eta = sp.Symbol("eta")
Nx, Ny = 5,4
vertices, cells = mesh(Nx, Ny)
dofmap = lambda e,r : cells[e,r]

def A_local(e):
    A_e = np.zeros((4,4))
    # Global coordinates for an element
    x = sum([vertices[dofmap(e,i)][0]*basis[i] for i in range(4)])
    y = sum([vertices[dofmap(e,i)][1]*basis[i] for i in range(4)])
    detDp = np.abs(x.diff("xi")*y.diff("eta")
                -x.diff("eta")*y.diff("xi"))

    quad_degree = 2
    points, weights = special.p_roots(quad_degree)

    for i in range(4):
        for j in range(4):
            dphi_i = sp.Matrix([[sp.diff(basis[i], "xi"),
                              sp.diff(basis[i], "eta")]])
            dphi_j = sp.Transpose(sp.Matrix([[sp.diff(basis[j], "xi"),
                                        sp.diff(basis[j], "eta")]]))
            phi_i = basis[i]
            phi_j = basis[j]

            grad_i = sp.Matrix([[sp.diff(basis[i], "xi")],
                                [sp.diff(basis[i], "eta")]])
            grad_j = sp.Matrix([[sp.diff(basis[j], "xi")],
                                [sp.diff(basis[j], "eta")]])
            Jac = sp.Matrix([[x.diff("xi"), x.diff("eta")],
                             [y.diff("xi"), y.diff("eta")]])
            for c_x in range(len(weights)):
                for c_y in range(len(weights)):
                    # Stiffness Matrix
                    Jac_loc = Jac.replace("xi", points[c_x]).replace("eta", points[c_y])

                    A_e[i,j] += weights[c_x]*weights[c_y]*detDp*(
                        (sp.transpose(Jac_loc.inv()*grad_j)*Jac_loc.inv()*grad_i).replace("xi", points[c_x]).
                        replace("eta", points[c_y]))[0,0]

                    #Mass matrix
                    # A_e[i,j] += weights[c_x]*weights[c_y]*detDp*(
                    #     (phi_i*phi_j).replace("xi", points[c_x]).
                    #     replace("eta", points[c_y]))
    return A_e

def MassMatrix():
    A = np.zeros((len(vertices),len(vertices)))
    for e in range(len(cells)):
        A_e = A_local(e)
        for r in range(4):
            for s in range(4):
                A[dofmap(e,r), dofmap(e,s)] += A_e[r,s]
    return A

def on_boundary():
    # Condition
    b_dofs = []
    cond = lambda x: (np.isclose(x[0],0) or np.isclose(x[1],0)
                        or np.isclose(x[0],1) or np.isclose(x[1],1))
    for i in range(len(vertices)):
        if cond(vertices[i]):
            b_dofs.append(i)
    return b_dofs

def bc_apply(A, b):
    b_dofs = on_boundary()
    for i in b_dofs:
        A[i,:i] = 0
        A[i,i+1:] =0
        A[i,i] = 1
        b[i] = 0
    return A

def b_local(e,f):
    B_e = np.zeros((4))
    # Global coordinates for an element
    x = sum([vertices[dofmap(e,i)][0]*basis[i] for i in range(4)])
    y = sum([vertices[dofmap(e,i)][1]*basis[i] for i in range(4)])
    detDp = np.abs(x.diff("xi")*y.diff("eta")
                -x.diff("eta")*y.diff("xi"))

    quad_degree = 3
    points, weights = special.p_roots(quad_degree)

    for i in range(4):
        phi_i = basis[i]
        for c_x in range(len(weights)):
            for c_y in range(len(weights)):
                B_e[i] += weights[c_x]*weights[c_y]*detDp*(
                    (f(x,y)*phi_i).replace("xi", points[c_x]).
                    replace("eta", points[c_y]))
    return B_e
    
def rhs(f):
    B = np.zeros(len(vertices))
    for e in range(len(cells)):
        B_e = b_local(e,f)
        for r in range(4):
            B[dofmap(e,r)] += B_e[r]
    return B


    
A= MassMatrix()
f = lambda x,y: 4*(-y**2+y)*sp.sin(sp.pi*x)
L = rhs(f)

bc_apply(A, L)
u_h = np.linalg.solve(A, L)


print("my solution")
#print(u_h)


# Reference

def J_local(e,coeffs):
    # Global coordinates for an element
    x = sum([vertices[dofmap(e,i)][0]*basis[i] for i in range(4)])
    y = sum([vertices[dofmap(e,i)][1]*basis[i] for i in range(4)])
    detDp = np.abs(x.diff("xi")*y.diff("eta")
                -x.diff("eta")*y.diff("xi"))

    quad_degree = 3
    points, weights = special.p_roots(quad_degree)
    loc = 0
    for i in range(4):
        for c_x in range(len(weights)):
            for c_y in range(len(weights)):
                loc +=  weights[c_x]*weights[c_y]*detDp*(
                    coeffs[dofmap(e,i)]*basis[i].replace("xi", points[c_x]).
                    replace("eta", points[c_y]))
    return loc
    
def J(u_h):
    value = 0
    for e in range(len(cells)):
        value += J_local(e,u_h)
    return value

print(J(u_h))

from dolfin import *
mesh = UnitSquareMesh.create(Nx,Ny, CellType.Type.quadrilateral)
V = FunctionSpace(mesh, "CG", 1)
u, v= TrialFunction(V), TestFunction(V)
a = inner(grad(u), grad(v))*dx
#a = inner(u,v)*dx
A = assemble(a)
x = SpatialCoordinate(mesh)
f = 4*(-x[1]**2+x[1])*sin(pi*x[0])
l = inner(f, v)*dx
B = assemble(l)
bc = DirichletBC(V, 0, "on_boundary")
bc.apply(A, B)
u_h = Function(V)
solve(A, u_h.vector(), B)
print("reference")
#print(u_h.vector().get_local())
print(assemble(u_h*dx))
