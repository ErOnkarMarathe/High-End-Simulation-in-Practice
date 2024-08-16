#include <iostream>
#include <fstream>
#include <cmath>
#include <iomanip>
#include <random>
#include <string>
#include <cuda_runtime.h>
#include <vector>

// Function to initialize particles
void initializeParticles(int numParticles, double* positions, double* velocities, double* masses, int gridSize, double spacing, double mass, double* domain) {
    std::mt19937 gen(42);
    std::normal_distribution<> d(0, 1);

    int count = 0;
    double offset = spacing / 2.0;
    double fluidspacing = spacing*2;
    double domainCenter[3] = { domain[0] / 2.0, domain[1] / 2.0, domain[2] / 2.0 };

    // Initialize particle positions and velocities
    for (int x = 0; x < gridSize; ++x) {
        for (int y = 0; y < gridSize; ++y) {
            for (int z = 0; z < gridSize; ++z) {
                if (count < numParticles) {
                    positions[count * 3 + 0] = x * fluidspacing + offset;
                    positions[count * 3 + 1] = y * fluidspacing + offset;
                    positions[count * 3 + 2] = z * fluidspacing + offset;

                    // Ensure particles are within the domain boundaries
                    positions[count * 3 + 0] = fmod(positions[count * 3 + 0], domain[0] - 2 * offset) + offset;
                    positions[count * 3 + 1] = fmod(positions[count * 3 + 1], domain[1] - 2 * offset) + offset;
                    positions[count * 3 + 2] = fmod(positions[count * 3 + 2], domain[2] - 2 * offset) + offset;

                    velocities[count * 3 + 0] = d(gen);
                    velocities[count * 3 + 1] = d(gen);
                    velocities[count * 3 + 2] = d(gen);

                    masses[count] = mass;
                    ++count;
                }
            }
        }
    }

    // Calculate the center of the grid
    double gridCenter[3] = {
        (gridSize - 1) * fluidspacing / 2.0 + offset,
        (gridSize - 1) * fluidspacing / 2.0 + offset,
        (gridSize - 1) * fluidspacing / 2.0 + offset
    };

    // Shift particles to center them within the domain
    for (int i = 0; i < numParticles; ++i) {
        positions[i * 3 + 0] += domainCenter[0] - gridCenter[0];
        positions[i * 3 + 1] += domainCenter[1] - gridCenter[1];
        positions[i * 3 + 2] += domainCenter[2] - gridCenter[2];
    }
}

// Function to initialize ghost particles around the domain
void initializeGhostParticles(double* ghostPositions, double* domain, double spacing, int factor, double mass, double* ghostMasses, double* ghostVelocities, int numGhostParticles) {
    int count = 0;
    double offset = 0;  // Start from 0 for the inner layer

    // Create ghost particles along the faces of the domain, outside its boundaries
    // Loop through each layer
    for (int layer = 1; layer <= factor; ++layer) {
        offset = layer * spacing;  // Update offset for each layer

        // X faces
        for (int y = 1; y <= domain[1] / spacing; ++y) {
            for (int z = 1; z <= domain[2] / spacing; ++z) {
                // Left face
                if (count < numGhostParticles) {
                    ghostPositions[count * 3 + 0] = -offset/2;
                    ghostPositions[count * 3 + 1] = (y -1)* spacing+(spacing/2);
                    ghostPositions[count * 3 + 2] = (z -1)* spacing+(spacing/2);
                    ghostVelocities[count * 3 + 0] = 0;
                    ghostVelocities[count * 3 + 1] = 0;
                    ghostVelocities[count * 3 + 2] = 0;
                    ghostMasses[count] = mass;
                    ++count;
                }

                // Right face
                if (count < numGhostParticles) {
                    ghostPositions[count * 3 + 0] = domain[0] + offset/2;
                    ghostPositions[count * 3 + 1] = (y -1)* spacing+(spacing/2);
                    ghostPositions[count * 3 + 2] = (z-1) * spacing+(spacing/2);
                    ghostVelocities[count * 3 + 0] = 0;
                    ghostVelocities[count * 3 + 1] = 0;
                    ghostVelocities[count * 3 + 2] = 0;
                    ghostMasses[count] = mass;
                    ++count;
                }
            }
        }

        // Y faces
        for (int x = 1; x <= domain[0] / spacing; ++x) {
            for (int z = 1; z <= domain[2] / spacing; ++z) {
                // Bottom face
                if (count < numGhostParticles) {
                    ghostPositions[count * 3 + 0] = (x-1) * spacing+(spacing/2);
                    ghostPositions[count * 3 + 1] = -offset/2;
                    ghostPositions[count * 3 + 2] = (z-1) * spacing+(spacing/2);
                    ghostVelocities[count * 3 + 0] = 0;
                    ghostVelocities[count * 3 + 1] = 0;
                    ghostVelocities[count * 3 + 2] = 0;
                    ghostMasses[count] = mass;
                    ++count;
                }

                // Top face
                if (count < numGhostParticles) {
                    ghostPositions[count * 3 + 0] = (x -1)* spacing +(spacing/2);
                    ghostPositions[count * 3 + 1] = domain[1] + offset/2;
                    ghostPositions[count * 3 + 2] = (z -1)* spacing +(spacing/2);
                    ghostVelocities[count * 3 + 0] = 0;
                    ghostVelocities[count * 3 + 1] = 0;
                    ghostVelocities[count * 3 + 2] = 0;
                    ghostMasses[count] = mass;
                    ++count;
                }
            }
        }

        // Z faces
        for (int x = 1; x <= domain[0] / spacing; ++x) {
            for (int y = 1; y <= domain[1] / spacing; ++y) {
                // Front face
                if (count < numGhostParticles) {
                    ghostPositions[count * 3 + 0] = (x-1) * spacing+(spacing/2);
                    ghostPositions[count * 3 + 1] = (y-1) * spacing+(spacing/2);
                    ghostPositions[count * 3 + 2] = -offset/2;
                    ghostVelocities[count * 3 + 0] = 0;
                    ghostVelocities[count * 3 + 1] = 0;
                    ghostVelocities[count * 3 + 2] = 0;
                    ghostMasses[count] = mass;
                    ++count;
                }

                // Back face
                if (count < numGhostParticles) {
                    ghostPositions[count * 3 + 0] = (x-1) * spacing +(spacing/2);
                    ghostPositions[count * 3 + 1] = (y-1) * spacing +(spacing/2);
                    ghostPositions[count * 3 + 2] = domain[2] + offset/2;
                    ghostVelocities[count * 3 + 0] = 0;
                    ghostVelocities[count * 3 + 1] = 0;
                    ghostVelocities[count * 3 + 2] = 0;
                    ghostMasses[count] = mass;
                    ++count;
                }
            }
        }
    }
}


// Function to write particles as spheres in VTK format
void writeParticles(std::ofstream &vtkFile, int numParticles,int numGhostParticles, double* positions,double * ghostPositions, double radius, double* velocities, double* densities) {
    vtkFile << "POINTS " << numParticles  << " float\n";
    for (int i = 0; i < numParticles; ++i) {
        vtkFile << positions[i * 3 + 0] << " " << positions[i * 3 + 1] << " " << positions[i * 3 + 2] << "\n";
    }
    //for (int i = 0; i < numGhostParticles; ++i) {
    //    vtkFile << ghostPositions[i * 3 + 0] << " " << ghostPositions[i * 3 + 1] << " " << ghostPositions[i * 3 + 2] << "\n";
    //}

    vtkFile << "\nPOINT_DATA " << numParticles << "\n";
    vtkFile << "SCALARS radius float 1\n";
    vtkFile << "LOOKUP_TABLE default\n";
    for (int i = 0; i < numParticles; ++i) {
        vtkFile << radius << "\n";
    }
    //for (int i = 0; i < numGhostParticles; ++i) {
    //    vtkFile << radius << "\n";
    //}
    vtkFile << "\nSCALARS density float 1\n";
    vtkFile << "LOOKUP_TABLE default\n";
    for (int i = 0; i < numParticles; ++i) {
        vtkFile << densities[i] << "\n";
    }
    vtkFile << "\nVECTORS velocity float\n";
    for (int i = 0; i < numParticles; ++i) {
        vtkFile << velocities[i * 3 + 0] << " " << velocities[i * 3 + 1] << " " << velocities[i * 3 + 2] << "\n";
    }
}


// CUDA Kernel function for cubic spline kernel
__device__ double W(double r, double h) {
    if (r >= 0 && r <= h) {
        return (315.0 / (64.0 * (22/7) * pow(h, 9))) * pow(h * h - r * r, 3);
    } else 
    {
        return 0.0;
    }
}

__device__ void Wp(double dx, double dy, double dz, double h, double& grad_Wp_x, double& grad_Wp_y, double& grad_Wp_z) {
    double r2 = dx * dx + dy * dy + dz * dz; // r^2
    if (r2 > 0 && r2 <= h * h) {
        double r_mag = sqrt(r2); // Calculate |r|
        double coeff = -(45.0 / ((22/7) * pow(h, 6))) * pow(h - r_mag, 2) / r_mag;
        grad_Wp_x = coeff * dx;
        grad_Wp_y = coeff * dy;
        grad_Wp_z = coeff * dz;
    } 
    else {
        grad_Wp_x = grad_Wp_y = grad_Wp_z = 0.0;
    }
}


__device__ double Wv(double r, double h) {
    if (r >= 0 && r <= h) {
        return (45.0 / ((22/7) * pow(h, 6))) * (h - r);
    } else {
        return 0.0;
    }
}


// CUDA Kernel function to calculate distance between two points in 3D
__device__ double distance(double* pos1, double* pos2) {
    return sqrt(
        pow(pos1[0] - pos2[0], 2) +
        pow(pos1[1] - pos2[1], 2) +
        pow(pos1[2] - pos2[2], 2)
    );
}

// CUDA Kernel function for density calculation
__global__ void calculateDensity(int numParticles, int numGhostParticles, 
                                  double* positions, double* densities, 
                                  double* masses, double* ghostPositions, 
                                  double* ghostDensities, double* ghostMasses, 
                                  double h, double rho_0) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < numParticles) {
        densities[i] = 0.0;  // Reset density for fluid particle i

        // From fluid particles
        for (int j = 0; j < numParticles; ++j) {
            double r = distance(&positions[i * 3], &positions[j * 3]);
            if (r <= h) {  // Only include contributions from particles within distance h
                densities[i] += masses[j] * W(r, h);
            }
        }
        
        // From ghost particles
        for (int j = 0; j < numGhostParticles; ++j) {
            double r = distance(&positions[i * 3], &ghostPositions[j * 3]);
            if (r <= h) {  // Only include contributions from ghost particles within distance h
                densities[i] += ghostMasses[j] * W(r, h);
            }
        }
    }
    // Initialize ghost densities
    if (i < numGhostParticles) {
        ghostDensities[i] = rho_0;
    }
}


// CUDA Kernel function for pressure calculation
__global__ void calculatePressure(int numParticles, int numGhostParticles, 
                                   double* densities, double* pressures, 
                                   double rho_0, double k, 
                                   double* ghostDensities, double* ghostPressures) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    // Pressure calculation for fluid particles
    if (i < numParticles) {
        pressures[i] = (densities[i] - rho_0) * k;
    }  
    ghostPressures[i] = 0.0;                                 
}

struct Cell {
    int head;
};

struct ParticleNode {
    int particleIndex;
    int next;
};

__global__ void initializeCells(int numCells, Cell* cells, ParticleNode* nodes, int numParticles) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < numCells) {
        cells[i].head = -1;
    }
    if (i < numParticles) {
        nodes[i].particleIndex = -1;
        nodes[i].next = -1;
    }
}

__global__ void linkedlist(int numParticles, double* positions, Cell* cells, ParticleNode* nodes, double* cellSize, double* domain) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < numParticles) {
        // Compute the cell index for the particle
        int cellX = (int)(positions[i * 3 + 0] / cellSize[0]);
        int cellY = (int)(positions[i * 3 + 1] / cellSize[1]);
        int cellZ = (int)(positions[i * 3 + 2] / cellSize[2]);

        // Ensure the particle is within the grid boundaries
        cellX = min(max(cellX, 0), (int)(domain[0] / cellSize[0]) - 1);
        cellY = min(max(cellY, 0), (int)(domain[1] / cellSize[1]) - 1);
        cellZ = min(max(cellZ, 0), (int)(domain[2] / cellSize[2]) - 1);

        int cellIndex = cellZ * (int)(domain[0] / cellSize[0]) * (int)(domain[1] / cellSize[1]) + cellY * (int)(domain[0] / cellSize[0]) + cellX;

        // Insert the particle into the linked list for the cell
        nodes[i].particleIndex = i;
        nodes[i].next = atomicExch(&cells[cellIndex].head, i);
    }
    __syncthreads();
}

__global__ void calculateForceslinkedlist(int numParticles, double* masses, double* positions, 
                                 double* velocities, double* densities, double* forces, 
                                 double* pressures, double h, double rho_0, 
                                 double mu, double g, double* ghostMasses, double* ghostPositions, 
                                 double* ghostVelocities, double* ghostDensities, 
                                 double* ghostPressures, int numGhostParticles, double radius,
                                 Cell* cells, ParticleNode* nodes, double* cellSize, double* domain) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < numParticles) {
        // Initialize forces
        double fgravity[3] = {0.0, -masses[i] * g, 0.0}; // gravity acts downwards in y-axis
        double fpressure[3] = {0.0, 0.0, 0.0};
        double fviscous[3] = {0.0, 0.0, 0.0};
        double fghostpressure[3] = {0.0, 0.0, 0.0};

        // Compute the cell index for the particle
        int cellX = (int)(positions[i * 3 + 0] / cellSize[0]);
        int cellY = (int)(positions[i * 3 + 1] / cellSize[1]);
        int cellZ = (int)(positions[i * 3 + 2] / cellSize[2]);

        // Ensure the particle is within the grid boundaries
        cellX = min(max(cellX, 0), (int)(domain[0] / cellSize[0]) - 1);
        cellY = min(max(cellY, 0), (int)(domain[1] / cellSize[1]) - 1);
        cellZ = min(max(cellZ, 0), (int)(domain[2] / cellSize[2]) - 1);

        // Iterate over neighboring cells
        for (int dx = -1; dx <= 1; ++dx) {
            for (int dy = -1; dy <= 1; ++dy) {
                for (int dz = -1; dz <= 1; ++dz) {
                    int neighborCellX = cellX + dx;
                    int neighborCellY = cellY + dy;
                    int neighborCellZ = cellZ + dz;

                    // Ensure neighboring cell is within the grid boundaries
                    if (neighborCellX >= 0 && neighborCellX < (int)(domain[0] / cellSize[0]) &&
                        neighborCellY >= 0 && neighborCellY < (int)(domain[1] / cellSize[1]) &&
                        neighborCellZ >= 0 && neighborCellZ < (int)(domain[2] / cellSize[2])) {

                        int neighborCellIndex = neighborCellZ * (int)(domain[0] / cellSize[0]) * (int)(domain[1] / cellSize[1]) + neighborCellY * (int)(domain[0] / cellSize[0]) + neighborCellX;

                        int particleIdx = cells[neighborCellIndex].head;
                        while (particleIdx != -1) {
                            if (particleIdx != i) {
                                // Calculate distance vector
                                double dx = positions[particleIdx * 3 + 0] - positions[i * 3 + 0];
                                double dy = positions[particleIdx * 3 + 1] - positions[i * 3 + 1];
                                double dz = positions[particleIdx * 3 + 2] - positions[i * 3 + 2];
                                double r_mag = sqrt(dx * dx + dy * dy + dz * dz);

                                if (r_mag < h) {
                                    // Pressure force contribution
                                    double grad_Wp_x, grad_Wp_y, grad_Wp_z;
                                    Wp(dx, dy, dz, h, grad_Wp_x, grad_Wp_y, grad_Wp_z);
                                    double pressure_term = -((pressures[i] - pressures[particleIdx]) / (2.0 * densities[particleIdx] + 0.01));
                                    fpressure[0] += masses[particleIdx] * pressure_term * grad_Wp_x;
                                    fpressure[1] += masses[particleIdx] * pressure_term * grad_Wp_y;
                                    fpressure[2] += masses[particleIdx] * pressure_term * grad_Wp_z;

                                    // Viscous force contribution
                                    double viscosity_term = (mu / rho_0) * Wv(r_mag, h);
                                    fviscous[0] += masses[particleIdx] * (velocities[particleIdx * 3 + 0] - velocities[i * 3 + 0]) * viscosity_term / (densities[particleIdx] + 0.01);
                                    fviscous[1] += masses[particleIdx] * (velocities[particleIdx * 3 + 1] - velocities[i * 3 + 1]) * viscosity_term / (densities[particleIdx] + 0.01);
                                    fviscous[2] += masses[particleIdx] * (velocities[particleIdx * 3 + 2] - velocities[i * 3 + 2]) * viscosity_term / (densities[particleIdx] + 0.01);
                                }
                            }
                            particleIdx = nodes[particleIdx].next;
                        }
                    }
                }
            }
        }

        // Interaction with ghost particles
        for (int j = 0; j < numGhostParticles; ++j) {
            // Calculate distance vector
            double dx = ghostPositions[j * 3 + 0] - positions[i * 3 + 0];
            double dy = ghostPositions[j * 3 + 1] - positions[i * 3 + 1];  
            double dz = ghostPositions[j * 3 + 2] - positions[i * 3 + 2];
            double r_mag = sqrt(dx * dx + dy * dy + dz * dz);

            if (r_mag < h) {
                // Pressure force contribution
                double grad_Wp_x, grad_Wp_y, grad_Wp_z;
                Wp(dx, dy, dz, h, grad_Wp_x, grad_Wp_y, grad_Wp_z);
                double pressure_term = -((pressures[i] - ghostPressures[j]) / (2.0 * ghostDensities[j] + 0.01));
                fghostpressure[0] += ghostMasses[j] * pressure_term * grad_Wp_x;
                fghostpressure[1] += ghostMasses[j] * pressure_term * grad_Wp_y;
                fghostpressure[2] += ghostMasses[j] * pressure_term * grad_Wp_z;
            }
        }
        // Total force
        forces[i * 3 + 0] = fgravity[0] + fpressure[0] + fviscous[0] + fghostpressure[0] ;
        forces[i * 3 + 1] = fgravity[1] + fpressure[1] + fviscous[1] + fghostpressure[1] ;
        forces[i * 3 + 2] = fgravity[2] + fpressure[2] + fviscous[2] + fghostpressure[2] ;
    }
}



__global__ void updateParticles(int numParticles, double* positions, double* velocities, double* forces, double* ghostForces, double* densities, double dt, double* domain, double radius) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < numParticles) {
        // Calculate acceleration
        double acc[3] = {
            forces[i * 3 + 0] / densities[i],
            forces[i * 3 + 1] / densities[i],
            forces[i * 3 + 2] / densities[i]
        };

        // Update velocities using Explicit Euler integration
        velocities[i * 3 + 0] += acc[0] * dt;
        velocities[i * 3 + 1] += acc[1] * dt;
        velocities[i * 3 + 2] += acc[2] * dt;

        // Update positions using Explicit Euler integration
        positions[i * 3 + 0] += velocities[i * 3 + 0] * dt;
        positions[i * 3 + 1] += velocities[i * 3 + 1] * dt;
        positions[i * 3 + 2] += velocities[i * 3 + 2] * dt;
    }
}


int main() {
    int numParticles, factor;
    double mass, radius;
    double domain[3],cellSize[3];
    double g, dt, time, k, rho_0, mu, h;
    Cell* d_cells;
    ParticleNode* d_nodes;

    std::cout << "Enter the number of particles: ";
    std::cin >> numParticles;

    std::cout << "Enter the mass of each particle: ";
    std::cin >> mass;

    std::cout << "Enter the radius of particles: ";
    std::cin >> radius;

    std::cout << "Enter domain dimensions (x y z): ";
    std::cin >> domain[0] >> domain[1] >> domain[2];

    std::cout << "Enter cell dimensions (x y z): ";
    std::cin >> cellSize[0] >> cellSize[1] >> cellSize[2];
    int numCells = (domain[0] * domain[1] * domain[2]) / (cellSize[0] * cellSize[1] * cellSize[2]);
    std::cout << "Total number of cells: " << numCells << std::endl;

    std::cout << "Enter the number of layers of ghost particles: ";
    std::cin >> factor;

    std::cout << "Enter the gas constant: ";
    std::cin >> k;
    std::cout << "Enter the density: ";
    std::cin >> rho_0;
    std::cout << "Enter the viscosity: ";
    std::cin >> mu;
    std::cout << "Enter the smoothness factor: ";
    std::cin >> h;
    std::cout << "Enter gravity: ";
    std::cin >> g;
    std::cout << "Enter size of time steps: ";
    std::cin >> dt;
    std::cout << "Enter the end time for simulation: ";
    std::cin >> time;

    // Calculate surface area and number of ghost particles
    double surfaceArea = (2 * (domain[0] * domain[1] + domain[0] * domain[2] + domain[1] * domain[2]));
    int numGhostParticles = static_cast<int>(factor * (surfaceArea / (radius * radius)));

    // Memory allocations
    double* d_positions;
    double* d_velocities;
    double* d_forces;
    double* d_masses;
    double* d_domain;
    double* d_densities;
    double* d_pressures;
    double* d_ghostPositions;
    double* d_ghostMasses;
    double* d_ghostVelocities;
    double* d_ghostDensities;
    double* d_ghostPressures;
    double* d_ghostForces;
    double* d_cellSize;
    int* d_cellIndices;

    cudaMallocManaged(&d_positions, numParticles * 3 * sizeof(double));
    cudaMallocManaged(&d_velocities, numParticles * 3 * sizeof(double));
    cudaMallocManaged(&d_forces, numParticles * 3 * sizeof(double));
    cudaMallocManaged(&d_masses, numParticles * sizeof(double));
    cudaMallocManaged(&d_domain, 3 * sizeof(double));
    cudaMallocManaged(&d_densities, numParticles * sizeof(double));
    cudaMallocManaged(&d_pressures, numParticles * sizeof(double));
    cudaMallocManaged(&d_ghostPositions, numGhostParticles * 3 * sizeof(double));
    cudaMallocManaged(&d_ghostMasses, numGhostParticles * sizeof(double));
    cudaMallocManaged(&d_ghostVelocities, numGhostParticles * 3 * sizeof(double));
    cudaMallocManaged(&d_ghostDensities, numGhostParticles * sizeof(double));
    cudaMallocManaged(&d_ghostPressures, numGhostParticles * sizeof(double));
    cudaMallocManaged(&d_ghostForces, numGhostParticles * 3 * sizeof(double));
    cudaMallocManaged(&d_cells, numCells * sizeof(Cell));
    cudaMallocManaged(&d_nodes, numParticles * sizeof(ParticleNode));
    cudaMallocManaged(&d_cellSize, 3 * sizeof(double));
    cudaMallocManaged(&d_cellIndices, numParticles * sizeof(int));

    int gridSize = std::ceil(std::pow(numParticles, 1.0 / 3.0));
    double spacing = 1.0 * radius;

    cudaMemcpy(d_domain, domain, 3 * sizeof(double), cudaMemcpyHostToDevice);
    initializeParticles(numParticles, d_positions, d_velocities, d_masses, gridSize, spacing, mass, d_domain);
    initializeGhostParticles(d_ghostPositions, d_domain, spacing, factor, mass, d_ghostMasses, d_ghostVelocities, numGhostParticles);
    initializeCells<<<(numCells + numParticles + 255) / 256, 256>>>(numCells, d_cells, d_nodes, numParticles);

    cudaMemcpy(d_domain, domain, 3 * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_cellSize, cellSize, 3 * sizeof(double), cudaMemcpyHostToDevice);

    int numSteps = time / dt;

    int blockSize = 256;
    int numBlocks = (numParticles + blockSize - 1) / blockSize;

    float totalTimeForComputation = 0.0f;

    // Allocate host memory
    double* h_positions = new double[numParticles * 3];

    // Open initial VTK file
    std::ofstream vtkFile("particles_0.vtk");
    vtkFile << "# vtk DataFile Version 3.0\n";
    vtkFile << "Particle Simulation\n";
    vtkFile << "ASCII\n";
    vtkFile << "DATASET POLYDATA\n";
    vtkFile << "POINTS " << (numParticles + numGhostParticles) << " float\n";

    // Write initial particles
    writeParticles(vtkFile, numParticles, numGhostParticles, d_positions, d_ghostPositions, radius, d_velocities, d_densities);

    vtkFile.close();

    for (int step = 1; step <= numSteps; ++step) 
    {
        cudaEvent_t start, stop;
        float elapsedTime;
        cudaEventCreate(&start);
        cudaEventRecord(start, 0);

        // 2a. Calculate Density
        calculateDensity<<<numBlocks, blockSize>>>(numParticles, numGhostParticles, d_positions, d_densities, d_masses, d_ghostPositions, d_ghostDensities, d_ghostMasses, h, rho_0);
        cudaDeviceSynchronize();

        // 3a. Calculate Pressure
        calculatePressure<<<numBlocks, blockSize>>>(numParticles, numGhostParticles, d_densities, d_pressures, rho_0, k, d_ghostDensities, d_ghostPressures);
        cudaDeviceSynchronize();

        // 5. Calculate forces
        calculateForceslinkedlist<<<numBlocks, blockSize>>>(numParticles, d_masses, d_positions, d_velocities, d_masses, d_forces, d_pressures, h, rho_0, mu, g, d_ghostMasses, d_ghostPositions, d_ghostVelocities, d_ghostDensities, d_ghostPressures, numGhostParticles, radius, d_cells, d_nodes, d_cellSize, d_domain);
        cudaDeviceSynchronize();
        

        // 6. integration step
        updateParticles<<<numBlocks, blockSize>>>(numParticles, d_positions, d_velocities, d_forces, d_ghostForces, d_densities, dt, d_domain, radius);
        cudaDeviceSynchronize();

        cudaEventCreate(&stop);
        cudaEventRecord(stop, 0);
        cudaEventSynchronize(stop);

        cudaEventElapsedTime(&elapsedTime, start, stop);
        totalTimeForComputation += elapsedTime;

        if (step % 10 == 0) {
            // Write particles to a new VTK file every 100 steps
            vtkFile.open("particles_" + std::to_string(step) + ".vtk");
            vtkFile << "# vtk DataFile Version 3.0\n";
            vtkFile << "Particle Simulation\n";
            vtkFile << "ASCII\n";
            vtkFile << "DATASET POLYDATA\n";
            vtkFile << "POINTS " << (numParticles + numGhostParticles) << " float\n";

            writeParticles(vtkFile, numParticles, numGhostParticles, d_positions, d_ghostPositions, radius, d_velocities, d_densities);
            vtkFile.close();

            std::ofstream outFile("time.txt", std::ofstream::app);
            outFile << "Step " << step << " Time: " << elapsedTime << "ms" << std::endl;
            outFile.close();
        }
    }

    printf("Total time for computation is: %f ms\n", totalTimeForComputation);

    // Free unified memory
    cudaFree(d_positions);
    cudaFree(d_velocities);
    cudaFree(d_forces);
    cudaFree(d_masses);
    cudaFree(d_domain);
    cudaFree(d_pressures);
    cudaFree(d_densities);
    cudaFree(d_ghostPositions);
    cudaFree(d_ghostMasses);
    cudaFree(d_ghostVelocities);
    cudaFree(d_ghostDensities);
    cudaFree(d_ghostPressures);
    cudaFree(d_ghostForces);
    cudaFree(d_cells);
    cudaFree(d_nodes);
    cudaFree(d_cellIndices);

    return 0;
}

