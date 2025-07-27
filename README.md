# Demo
This repository contains various CUDA codes implementing different physical simulation scenarios:

1. Molecular Dynamics using Lennard-Jones forces over an infinite domain
Particles are initially distributed according to a Gaussian distribution. During the simulation, particles interact via Lennard-Jones forces. Time integration is performed using the Leapfrog method. The total energy of the system is monitored as a checkpoint to ensure it remains constant, validating energy conservation.

2. Molecular Dynamics using Lennard-Jones forces over a fixed domain 
   a. using Periodic bc
   b. using cut-off radius and periodic bc
   c. using cut-off radius and periodic bc with a neighbour list (accelerated version)
This scenario extends the first by constraining particles inside a fixed simulation box with periodic boundary conditions, allowing particles to seamlessly re-enter the domain. To optimise performance, a cut-off radius limits force calculations to nearby particles. The accelerated version utilizes a neighbor list implemented as a linked list data structure to reduce computational complexity.

3. Molecular Dynamics using Spring-Dashpot model 
This version models particle interactions using a spring-dashpot force model with reflexive boundary conditions applied on all domain walls. Time integration is performed using the Explicit Euler method.

4. Smoothed Particle Hydrodynamic
Implements SPH to simulate fluid particle behaviour, modelling fluid particles falling within a domain and interacting based on kernel smoothing functions.

# Additional Explanation for file naming
1. moleculardynamics.cu is file for 1st Task
2. PeriodicBC.cu, PeriodicBCcutoff.cu and PeriodicBCcutoffAccelerated.cu is for 2nd Task. 
3. DEMacceleratedversion.cu is for 3rd Task
4. sphacc.cu and sph.cu are for 4th Task.

And each MP4 file is a visualisation for each scenario.
