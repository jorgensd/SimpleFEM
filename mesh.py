import numpy as np
import sympy as sp
from itertools import product
from IPython import embed
from scipy import special

# Basis functions for Quadrilateral element
xi = sp.Symbol("xi")
eta = sp.Symbol("eta")
basis_functions = [(xi-1)*(eta-1)/4,
                   -(xi+1)*(eta-1)/4,
                   (xi+1)*(eta+1)/4,
                   -(xi-1)*(eta+1)/4]
    
class QuadElement():
    def __init__(self, nodes, glob_num):
        # Local to global dof-map
        self.nodes = nodes
        self.local_to_global = {i: glob_num[i] for i in range(4)}
        self.global_x = sum([nodes[i][0]*basis_functions[i] for i in range(4)])
        self.global_y = sum([nodes[i][1]*basis_functions[i] for i in range(4)])
        self.Dp = np.abs(self.global_x.diff("xi")*self.global_y.diff("eta")
                         - self.global_x.diff("eta")*self.global_y.diff("xi"))

class UnitSquareMesh():
    def __init__(self, Nx, Ny):
        xs = np.linspace(0,1,Nx+1)
        ys = np.linspace(0,1,Ny+1)
        self.dofs = [[xs[i],ys[j]] for i in range(Nx+1) for j in range(Ny+1)]
        self.elements = []
        for i in range(Nx*Ny):
            j = i//Ny
            self.elements.append(QuadElement([self.dofs[i+j], self.dofs[i+Ny+1+j],
                                              self.dofs[i+Ny+2+j],
                                              self.dofs[i+1+j]],
                                             [i+j,i+Ny+1+j,i+Ny+2+j,i+1+j]))    
        # for e in self.elements: 
        #     print(e.local_to_global[0],e.local_to_global[1],e.local_to_global[2],e.local_to_global[3])
        # exit(1)

quad_degree = 3
points, weights = special.p_roots(quad_degree)

def integrate(f, mesh):
    integral = 0
    for e in mesh.elements:
        for c_x in range(len(weights)):
            for c_y in range(len(weights)):
                local_x = e.global_x.replace("xi", points[c_x]).replace("eta", points[c_y])
                local_y = e.global_y.replace("xi", points[c_x]).replace("eta", points[c_y])
                integral += weights[c_x]*weights[c_y]*f(local_x,local_y)*e.Dp
    return float(integral)

import pytest

@pytest.mark.parametrize("Nx",[1,2,3,4,5])
@pytest.mark.parametrize("Ny",[1,2,3,4,5])
def test_integrate_one(Nx, Ny):
    mesh = UnitSquareMesh(Nx,Ny)
    e = mesh.elements[0]
    f = lambda x,y : 1# x*y**2

    intf = integrate(f, mesh)
    assert(np.isclose(intf,1))


@pytest.mark.parametrize("Nx",[1,2,3,4,5])
@pytest.mark.parametrize("Ny",[1,2,3,4,5])
def test_integrate_quad(Nx, Ny):
    mesh = UnitSquareMesh(Nx,Ny)
    e = mesh.elements[0]
    n = 2
    m = 4
    f = lambda x,y : x**n*y**m

    intf = integrate(f, mesh)
    print(abs(intf-1/6*1/3))
    assert(np.isclose(intf, 1/((n+1)*(m+1))))

