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
x00,x10,x11,x01 = sp.symbols("x00,x10,x11,x01")
y00,y10,y11,y01 = sp.symbols("y00,y10,y11,y01")
x_ = [x00,x10,x11,x01]
y_ = [y00,y10,y11,y01]
cx_basis = sum([x_[i]*basis[i] for i in range(4)])
cy_basis = sum([y_[i]*basis[i] for i in range(4)])
Jac = sp.Matrix([[cx_basis.diff(xi), cx_basis.diff(eta)],
                 [cy_basis.diff(xi), cy_basis.diff(eta)]])

grads = [sp.Matrix([[sp.diff(basis[i], "xi")],
                    [sp.diff(basis[i], "eta")]]) for i in range(4)]


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
    X, Y = np.meshgrid(x, y)
    grid = (X, Y)
    return vertices, cells, grid

xi = sp.Symbol("xi")
eta = sp.Symbol("eta")
Nx, Ny = 25,25
vertices, cells, grid = mesh(Nx, Ny)
dofmap = lambda e,r : cells[e,r]



def A_local(e, quad_degree=4):
    A_e = np.zeros((4,4))
    # Global coordinates for an element
    x_map = cx_basis.subs([(x_[i], vertices[dofmap(e,i)][0]) for i in range(4)])
    y_map = cy_basis.subs([(y_[i], vertices[dofmap(e,i)][1]) for i in range(4)])
    # Local Jacobian of determinant
    detJac_loc = Jac.det().subs([(x_[i], vertices[dofmap(e,i)][0])
                                 for i in range(4)])
    detJac_loc = detJac_loc.subs([(y_[i], vertices[dofmap(e,i)][1])
                                  for i in range(4)])
    # Local Jacobian
    Jac_loc = Jac.subs([(x_[i], vertices[dofmap(e,i)][0])
                        for i in range(4)])
    Jac_loc = Jac_loc.subs([(y_[i], vertices[dofmap(e,i)][1])
                             for i in range(4)])

    points, weights = special.p_roots(quad_degree)

    for i in range(4):
        for j in range(4):
            # Looping over quadrature points on ref element
            for c_x in range(len(weights)):
                for c_y in range(len(weights)):
                    # Stiffness Matrix
                    Jac_loc = Jac_loc.subs([("xi", points[c_x]),
                                            ("eta", points[c_y])])
                    A_e[i,j] += weights[c_x]*weights[c_y]*detJac_loc*(
                        (sp.transpose(Jac_loc.inv()*grads[j])
                         *Jac_loc.inv()*grads[i])
                        .subs([("xi", points[c_x]),("eta", points[c_y])])[0,0])

                    # #Mass matrix
                    # A_e[i,j] += weights[c_x]*weights[c_y]*detDp*(
                    #     (basis[i]*basis[j]).subs([("xi", points[c_x]),
                    #                               ("eta", points[c_y])])
    return A_e

def MassMatrix(quad_degree=4):
    A = np.zeros((len(vertices),len(vertices)))
    # All elements are the same
    A_e = A_local(0, quad_degree)
    for e in range(len(cells)):
        #A_e = A_local(e)
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

def b_local(e,f, quad_degree=4):
    B_e = np.zeros((4))

    # Global coordinates for an element
    x_map = cx_basis.subs([(x_[i], vertices[dofmap(e,i)][0]) for i in range(4)])
    y_map = cy_basis.subs([(y_[i], vertices[dofmap(e,i)][1]) for i in range(4)])
    # Local Jacobian of determinant
    detJac_loc = Jac.det().subs([(x_[i], vertices[dofmap(e,i)][0])
                                 for i in range(4)])
    detJac_loc = detJac_loc.subs([(y_[i], vertices[dofmap(e,i)][1])
                                  for i in range(4)])

    # Use Gauss-Legendre quadrature
    points, weights = special.p_roots(quad_degree)
    for i in range(4):
        for c_x in range(len(weights)):
            for c_y in range(len(weights)):
                B_e[i] += weights[c_x]*weights[c_y]*detJac_loc*(
                    (f(x_map,y_map)*basis[i]).subs([("xi", points[c_x]),
                                                    ("eta", points[c_y])]))
    return B_e
    
def rhs(f, quad_degree=4):
    B = np.zeros(len(vertices))
    for e in range(len(cells)):
        B_e = b_local(e,f, quad_degree)
        for r in range(4):
            B[dofmap(e,r)] += B_e[r]
    return B


import time;
start = time.time()
q_deg = 4
A= MassMatrix(q_deg)
stop = time.time()
print("Mass matrix assembly: {0:.2f}".format(stop-start))
f = lambda x,y: 4*(-y**2+y)*sp.sin(sp.pi*x)
start = time.time()
L = rhs(f,q_deg)
stop = time.time()
print("RHS assembly: {0:.2f}".format(stop-start))
bc_apply(A, L)
u_h = np.linalg.solve(A, L)


print("my solution")
#print(u_h)
import matplotlib.pyplot as plt
import matplotlib.collections as mplc
from sympy.plotting import plot3d
x_ , y_ = sp.symbols("x y")


# def plot(vertices, cells, u_h):
#     cell_vertex_coordinates = []
#     u_s = []
#     for i in range(len(cells)):
#         local_vertex_num = cells[i,:]
#         l_coor = vertices[local_vertex_num,:]
#         print(l_coor)
#         u_local = 0
#         for j in range(4):
#             u_local += u_h[local_vertex_num[j]]*basis[j]
#         x_min,x_max = min(l_coor[:,0]), max(l_coor[:,0])
#         y_min,y_max = min(l_coor[:,1]), max(l_coor[:,1])
#         print(x_min, x_max, y_min, y_max)
#         u_glob = u_local.replace("xi",
#                                  2*(x_-x_min)/(x_max-x_min))
#         u_glob = u_glob.replace("eta",
#                                 2*(y_-y_min)/(y_max-y_min))

#         u_s.append(u_glob)
#         exit(1)
#         cell_vertex_coordinates.append(l_coor)
#     tup = ()
#     for i in range(0,len(cells)):
#         tup = tup +(u_s[i],
#                          (x_,cell_vertex_coordinates[i][0,0],
#                           cell_vertex_coordinates[i][2,0]),
#                          (y_, cell_vertex_coordinates[0][0,1],
#                           cell_vertex_coordinates[i][2,1]),)
#     plot3d(*tup)
#     plt.savefig("fullpicture.png")

def plot(vertices, cells, u_h, grid):
    u_plot = u_h.reshape(grid[0].shape)
    plt.contourf(grid[0],grid[1], u_plot)
    # plt.imshow(u_plot, interpolation="bilinear", aspect="equal", origin="lower")
    for i in range(len(cells)):
        plt.plot(vertices[cells[i],0], vertices[cells[i], 1], "ko",alpha=0.5)
    plt.colorbar()
    plt.axis("equal")
    plt.savefig("result.png")

plot(vertices, cells, u_h,grid)

def J_local(e,coeffs, quad_degree=4):
    # Global coordinates for an element
    x = sum([vertices[dofmap(e,i)][0]*basis[i] for i in range(4)])
    y = sum([vertices[dofmap(e,i)][1]*basis[i] for i in range(4)])
    detDp = np.abs(x.diff("xi")*y.diff("eta")
                -x.diff("eta")*y.diff("xi"))

    points, weights = special.p_roots(quad_degree)
    loc = 0
    for i in range(4):
        for c_x in range(len(weights)):
            for c_y in range(len(weights)):
                loc +=  weights[c_x]*weights[c_y]*detDp*(
                    (coeffs[dofmap(e,i)]*basis[i])).subs([("xi", points[c_x]),
                                                          ("eta", points[c_y])])
    return loc
    
def J(u_h,quad_degree=4):
    value = 0
    for e in range(len(cells)):
        value += J_local(e,u_h,quad_degree)
    return value

print(J(u_h, 2))
import matplotlib.pyplot as plt


from dolfin import *
def dolfin_comparasion():
    mesh = UnitSquareMesh.create(Nx,Ny, CellType.Type.quadrilateral)
    V = FunctionSpace(mesh, "CG", 1)
    u, v= TrialFunction(V), TestFunction(V)
    a = inner(grad(u), grad(v))*dx
    #a = inner(u,v)*dx
    A_dol = assemble(a)
    x = SpatialCoordinate(mesh)
    f = 4*(-x[1]**2+x[1])*sin(pi*x[0])
    l = inner(f, v)*dx
    B_dol = assemble(l)
    bc = DirichletBC(V, 0, "on_boundary")
    bc.apply(A_dol, B_dol)
    u_h = Function(V)
    solve(A_dol, u_h.vector(), B_dol)
    print("reference")
    #print(u_h.vector().get_local())
    print(assemble(u_h*dx))
    File("u_h.pvd") << u_h

dolfin_comparasion()
