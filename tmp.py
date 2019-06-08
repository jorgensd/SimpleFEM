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

grad_basis = [[sp.diff(basis[i], xi), sp.diff(basis[i], eta)] for i in range(4)]
vertices, cells = mesh(1,1)
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
                    # A_e[i,j] += weights[c_x]*weights[c_y]*DetDp*(
                    #     (phi_i*phi_j).replace("xi", points[c_x]).
                    #     replace("eta", points[c_y]))
    return A_e

def A():
    A = np.zeros((len(vertices),len(vertices)))
    for e in range(len(cells)):
        A_e = A_local(e)
        for r in range(4):
            for s in range(4):
                A[dofmap(e,r), dofmap(e,s)] += A_e[r,s]
    print(A)
    embed()
A()
