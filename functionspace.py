import sympy as sp
from IPython import embed

class FunctionSpace():
    """
    Symbolic representation of the quadrilateral elements with four nodes
    """
    xi = sp.Symbol("xi")
    eta = sp.Symbol("eta")
    x_ = sp.symbols("x00,x10,x11,x01")
    y_ = sp.symbols("y00,y10,y11,y01")
    basis = sp.Matrix([(xi-1)*(eta-1)/4,
             -(xi+1)*(eta-1)/4,
             (xi+1)*(eta+1)/4,
             -(xi-1)*(eta+1)/4])
    # Global coordinate as function of reference parameterization
    c_basis = sp.Matrix([0,0])
    for i in range(4):
        c_basis[0] += x_[i]*basis[i]
        c_basis[1] += y_[i]*basis[i]
    Jac = c_basis.jacobian([xi, eta])
    grads = []
    for i in range(4):
        grads.append(sp.Matrix([[sp.diff(basis[i], xi)],
                                [sp.diff(basis[i], eta)]]))

    def __init__(self, mesh):
        self.mesh = mesh
        self._dofmap =  lambda e,r: self.mesh.cells[e,r]

    def dofmap(self):
        """
        Return dofmap for CG1 quad element, which is 
        the same the local to global vertex number mapping of the mesh
        """
        return self._dofmap
