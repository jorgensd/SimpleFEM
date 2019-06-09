# SimpleFEM
This repository contains a Finite Element Solver for the Poisson equation - &Delta; u = f on &Omega; = [0,1]&#215;[0,1],
a unit square consisting of N<sub>x</sub> elements in the x direction, N<sub>y</sub> elements in the y-direction.

## Structure:

- assembly.py: Contains assembly routines for the Stiffness matrix, the right hand side and integration of a single finite 
element function. 
- functionspace.py: Contains the implementation of a minimal version of the function space for
first order Lagrange elements on quadrilaterals, corresponding to having dofs at each vertex.
- mesh.py: Contains the implementation of the UnitSquareMesh
- problem.py: Contains the example where f=4(-y<sup>2</sup>+y) sin(&#960; x).
- test_solution.py: Compares &#8747; u dx obtained with this finite element solver a FEniCS implementation.
- plotting.py: Contains several plotting routines for visualising the solution.

![Solution of the Poisson equation visualized](u_h.png)
