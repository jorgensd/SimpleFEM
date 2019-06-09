from problem import main
from dolfin_reference import dolfin_comparasion
from assembly import assemble_volume_function

import pytest
import numpy as np


@pytest.mark.parametrize("Nx", [8, 20])
@pytest.mark.parametrize("Ny", [8, 15])
@pytest.mark.parametrize("f", ["4*(-y**2+y)*sin(pi*x)",
                               "8*pi**2*sin(2*pi*x)*sin(2*pi*y)",
                               "x**2+cos(2*pi*y)"])
def test(Nx, Ny, f):
    def f_internal(x, y):
        from sympy import pi, sin, cos
        return eval(f)
    q = 3
    u_h, V = main(Nx, Ny, f_internal, quad_degree=q)
    integral_u = assemble_volume_function(u_h, V, quad_degree=q)
    u_hd, u_dolfin = dolfin_comparasion(Nx, Ny, f)
    assert(np.isclose(np.linalg.norm(u_h),
                      np.linalg.norm(u_hd.vector().get_local())))
    assert(np.isclose(float(integral_u), u_dolfin))
