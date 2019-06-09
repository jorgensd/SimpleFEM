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

## Dependencies
This finite element solver uses the numpy, sympy, scipy and matplotlib library.
To run the comparasion of results with dolfin, you can use the FEniCS docker image, running
```
docker run --rm -ti -v $(pwd):/home/fenics/shared/ -w /home/fenics/shared/ quay.io/fenicsproject/stable:latest
```

## Results

### Solution of the Poisson equation on a 25&#215;10 grid with f=4(-y<sup>2</sup>+y) sin(&#960; x)
![Solution of the Poisson equation visualized](u_h.png)

### Visualization of the solution from a 25&#215;10 with the exact values on a 40&#215;40 grid.
![Custom mesh visualization](u_h_custom.png)

### Contour-plot of the solution from a 25&#215;10 grid.
![Custom mesh visualization](u_h_contour.png)
