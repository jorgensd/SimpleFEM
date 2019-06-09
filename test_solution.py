from problem import main
from dolfin_reference import dolfin_comparasion
import pytest
import numpy as np


@pytest.mark.parametrize("Nx", [8, 20])
@pytest.mark.parametrize("Ny", [8, 15])
def test(Nx, Ny):
    u_h, u_int = main(Nx, Ny)
    u_hd, u_dolfin = dolfin_comparasion(Nx, Ny)
    print(u_int, u_dolfin)
    assert(np.isclose(u_int, u_dolfin, rtol=1e-8))
