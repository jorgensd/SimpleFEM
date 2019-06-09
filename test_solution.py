from problem import main
from dolfin_reference import dolfin_comparasion
from assembly import assemble_volume_function

import pytest
import numpy as np


@pytest.mark.parametrize("Nx", [8, 20, 40])
@pytest.mark.parametrize("Ny", [8, 15, 33])
@pytest.mark.parametrize("f", ["4*(-y**2+y)*sin(pi*x)",
                               "8*pi**2*sin(2*pi*x)*sin(2*pi*y)",
                               "x**2+cos(2*pi*y)"])
def test(Nx, Ny, f):
    def f_internal(x, y):
        from sympy import pi, sin, cos
        return eval(f)
    u_h, V = main(Nx, Ny, f_internal)
    integral_u = assemble_volume_function(u_h, V, quad_degree=4)
    u_hd, u_dolfin = dolfin_comparasion(Nx, Ny, f)
    print(integral_u, u_dolfin)
    assert(np.isclose(float(integral_u), u_dolfin, rtol=1e-8))
