from mesh import UnitSquareMesh
from functionspace import FunctionSpace
from assembly import assemble_stiffness_matrix, assemble_rhs
from IPython import embed

mesh = UnitSquareMesh(2,2)
V = FunctionSpace(mesh)

A = assemble_stiffness_matrix(V)
import sympy as sp
x, y = sp.symbols("x y")
f = lambda x,y: 4*(-y**2+y)*sp.sin(sp.pi*x)
L = assemble_rhs(V, f)
embed()
