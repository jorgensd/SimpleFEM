from scipy import special
import numpy as np
import sympy as sp
from IPython import embed

def assemble_stiffness_matrix(V, quad_degree=4):
    """
    Assemble inner(grad(u),grad(v))*dx using the Gauss-Legendre quadrature.
    """
    N = len(V.mesh.vertices)
    A = np.zeros((N,N))
    dofmap  = V.dofmap()
    # All elements are the same
    A_e = A_local(V, 0, quad_degree)
    for e in range(len(V.mesh.cells)):
        # For other kinds of elements one would have to a separate A_e
        # for each element (or if the cell-sizes of the mesh would have changed.
        #A_e = A_local(e)
        for r in range(4):
            for s in range(4):
                A[dofmap(e,r), dofmap(e,s)] += A_e[r,s]
    print(A)
    return A

def A_local(V, e, quad_degree=4):
    """
    Assemble the local stiffness matrix on element # e
    """
    vertices = V.mesh.vertices
    dofmap = V.dofmap()
    A_e = np.zeros((4,4))
    # Local Jacobian of determinant
    detJac_loc = V.Jac.det().subs([(V.x_[i], vertices[dofmap(e,i)][0])
                                 for i in range(4)])
    detJac_loc = detJac_loc.subs([(V.y_[i], vertices[dofmap(e,i)][1])
                                  for i in range(4)])
    # Local Jacobian
    Jac_loc = V.Jac.subs([(V.x_[i], vertices[dofmap(e,i)][0])
                        for i in range(4)])
    Jac_loc = Jac_loc.subs([(V.y_[i], vertices[dofmap(e,i)][1])
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
                        (sp.transpose(Jac_loc.inv()*V.grads[j])
                         *Jac_loc.inv()*V.grads[i])
                        .subs([("xi", points[c_x]),("eta", points[c_y])])[0,0])
    return A_e

def assemble_rhs(V,f, quad_degree=4):
    """
    Assemble inner(f,v)*dx
    Input:
         V: Functionspace
         f: lambda f (x,y) : return ...
    """
    B = np.zeros(len(V.mesh.vertices))
    # Loop over each local cel
    for e in range(len(V.mesh.cells)):
        # Compute integral over reference element
        B_e = source_local(V, e, f, quad_degree)
        # Put local contributions in global rhs
        for r in range(4):
            B[V.dofmap()(e,r)] += B_e[r]
    return B

def source_local(V, e, f, quad_degree=4):
    B_e = np.zeros((4))
    c_basis = V.c_basis
    vertices = V.mesh.vertices
    dofmap = V.dofmap()
    # Global coordinates for an element
    x_map = c_basis[0].subs([(V.x_[i], vertices[dofmap(e,i)][0])
                             for i in range(4)])
    y_map = c_basis[1].subs([(V.y_[i], vertices[dofmap(e,i)][1])
                             for i in range(4)])
    
    # Local Jacobian of determinant
    detJac_loc = V.Jac.det().subs([(V.x_[i], vertices[dofmap(e,i)][0])
                                 for i in range(4)])
    detJac_loc = detJac_loc.subs([(V.y_[i], vertices[dofmap(e,i)][1])
                                  for i in range(4)])

    # Use Gauss-Legendre quadrature
    points, weights = special.p_roots(quad_degree)
    for i in range(4):
        for c_x in range(len(weights)):
            for c_y in range(len(weights)):
                B_e[i] += weights[c_x]*weights[c_y]*detJac_loc*(
                    (f(x_map,y_map)*V.basis[i]).subs([("xi", points[c_x]),
                                                    ("eta", points[c_y])]))
    return B_e


def volume_local(V,e,coeffs, quad_degree=4):
    dofmap = V.dofmap()
    # Global coordinates for an element
    x_map = V.c_basis[0].subs([(V.x_[i], V.mesh.vertices[dofmap(e,i)][0])
                             for i in range(4)])
    y_map = V.c_basis[1].subs([(V.y_[i], V.mesh.vertices[dofmap(e,i)][1])
                             for i in range(4)])
    
    # Local Jacobian of determinant
    detJac_loc = V.Jac.det().subs([(V.x_[i], V.mesh.vertices[dofmap(e,i)][0])
                                 for i in range(4)])
    detJac_loc = detJac_loc.subs([(V.y_[i], V.mesh.vertices[dofmap(e,i)][1])
                                  for i in range(4)])
  
    points, weights = special.p_roots(quad_degree)
    loc = 0
    for i in range(4):
        for c_x in range(len(weights)):
            for c_y in range(len(weights)):
                loc +=  weights[c_x]*weights[c_y]*\
                (detJac_loc*((coeffs[dofmap(e,i)]*V.basis[i]))
                 .subs([("xi", points[c_x]),("eta", points[c_y])]))
    return loc
    

def assemble_volume_function(u_h,V,quad_degree=4):
    """
    Assembles u_h*dx using Gauss-legendre quadrature rules
    Input:
        u_h: list containing the coefficient values for the
        finite element function.
        V: The finite element function space
    """
    value = 0
    for e in range(len(V.mesh.cells)):
        value += volume_local(V,e,u_h,quad_degree)
    return value
