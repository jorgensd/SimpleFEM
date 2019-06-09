from problem import main
from dolfin_reference import dolfin_comparasion
import pytest
import numpy as np


@pytest.mark.parametrize("Nx", [8, 20])
@pytest.mark.parametrize("Ny", [8, 15])
@pytest.mark.parametrize("f", ["4*(-y**2+y)*sin(pi*x)",
                               "8*pi**2*sin(2*pi*x)*sin(2*pi*y)"])
def test(Nx, Ny, f):
    def f_internal(x, y):
        from sympy import pi, sin, cos
        return eval(f)
    u_h, u_int = main(Nx, Ny, f_internal)
    u_hd, u_dolfin = dolfin_comparasion(Nx, Ny,f)
    print(u_int, u_dolfin)
    assert(np.isclose(u_int, u_dolfin, rtol=1e-8))
