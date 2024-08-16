#include <iostream>
#include <fstream>
#include <cmath>
#include <iomanip>
#include <random>
#include <string>
#include <cuda_runtime.h>


// Custom atomicAdd for double precision
__device__ double atomicAddDouble(double* address, double val) {
    unsigned long long int* address_as_ull = (unsigned long long int*)address;
    unsigned long long int old = *address_as_ull, assumed;
    do {
        assumed = old;
        old = atomicCAS(address_as_ull, assumed,
                        __double_as_longlong(val + __longlong_as_double(assumed)));
    } while (assumed != old);
    return __longlong_as_double(old);
}

__global__ void calculateForces(int numParticles, double* positions, double* velocities, double* forces, double* masses, double g, double K, double dampingFactor, double radius, double* domain) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < numParticles) {
        // Initialize force to zero
        double force[3] = {0.0, 0.0, 0.0};

        // Apply gravitational force (mg) in the negative y-direction
        force[1] = -masses[i] * g;

        // Iterate over all particle pairs to calculate interaction forces
        for (int j = 0; j < numParticles; j++) {
            if (i != j) {
                // Calculate the distance vector between particles i and j
                double xij[3] = {
                    positions[j * 3 + 0] - positions[i * 3 + 0],
                    positions[j * 3 + 1] - positions[i * 3 + 1],
                    positions[j * 3 + 2] - positions[i * 3 + 2]
                };

                // Calculate the distance magnitude
                double distance = sqrt(xij[0] * xij[0] + xij[1] * xij[1] + xij[2] * xij[2]);
                double overlap = 2 * radius - distance;

                // Check if particles are overlapping
                if (overlap > 0.0) {
                    // Calculate the normalized distance vector
                    double xij_norm[3] = {
                        xij[0] / distance,
                        xij[1] / distance,
                        xij[2] / distance
                    };

                    // Calculate the relative velocity between particles i and j
                    double vij[3] = {
                        velocities[j * 3 + 0] - velocities[i * 3 + 0],
                        velocities[j * 3 + 1] - velocities[i * 3 + 1],
                        velocities[j * 3 + 2] - velocities[i * 3 + 2]
                    };

                    // Calculate the spring force component
                    double springForce = K * overlap * overlap;

                    // Calculate the dashpot force component
                    double vij_dot_xij = vij[0] * xij_norm[0] + vij[1] * xij_norm[1] + vij[2] * xij_norm[2];  // Projection of vij on xij
                    double dashpotForce = -dampingFactor * vij_dot_xij;

                    // Calculate the total force between particles i and j
                    double FSD[3] = {
                        (springForce + dashpotForce) * xij_norm[0],
                        (springForce + dashpotForce) * xij_norm[1],
                        (springForce + dashpotForce) * xij_norm[2]
                    };

                    // Add the force to particle i
                    force[0] += FSD[0];
                    force[1] += FSD[1];
                    force[2] += FSD[2];
                }
            }
        }

        // Wall interactions using spring-dashpot model

        // Check x boundaries
        if (positions[i * 3 + 0] == radius) {
            double springForce = K * radius * radius;
            double dashpotForce = -dampingFactor * velocities[i * 3 + 0];
            force[0] += springForce + dashpotForce;
        }
        if (positions[i * 3 + 0] == domain[0] - radius) {
            double springForce = K * radius * radius;
            double dashpotForce = dampingFactor * velocities[i * 3 + 0];
            force[0] -= springForce + dashpotForce;
        }

        // Check y boundaries
        if (positions[i * 3 + 1] == radius) {
            double springForce = K * radius * radius;
            double dashpotForce = -dampingFactor * velocities[i * 3 + 1];
            force[1] += springForce + dashpotForce;
        }
        if (positions[i * 3 + 1] == domain[1] - radius) {
            double springForce = K * radius * radius;
            double dashpotForce = dampingFactor * velocities[i * 3 + 1];
            force[1] -= springForce + dashpotForce;
        }

        // Check z boundaries
        if (positions[i * 3 + 2] == radius) {
            double springForce = K * radius * radius;
            double dashpotForce = -dampingFactor * velocities[i * 3 + 2];
            force[2] += springForce + dashpotForce;
        }
        if (positions[i * 3 + 2] == domain[2] - radius) {
            double springForce = K * radius * radius;
            double dashpotForce = dampingFactor * velocities[i * 3 + 2];
            force[2] -= springForce + dashpotForce;
        }

        // Store the computed force for particle i
        forces[i * 3 + 0] = force[0];
        forces[i * 3 + 1] = force[1];
        forces[i * 3 + 2] = force[2];
    }
}



__global__ void updateParticles(int numParticles, double* positions, double* velocities, double* forces, double* masses, double dt, double* domain) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < numParticles) {
        // Calculate acceleration
        double acc[3] = {
            forces[i * 3 + 0] / masses[i],
            forces[i * 3 + 1] / masses[i],
            forces[i * 3 + 2] / masses[i]
        };

        // Update velocities using Explicit Euler integration
        velocities[i * 3 + 0] += acc[0] * dt;
        velocities[i * 3 + 1] += acc[1] * dt;
        velocities[i * 3 + 2] += acc[2] * dt;

        // Update positions using Explicit Euler integration
        positions[i * 3 + 0] += velocities[i * 3 + 0] * dt;
        positions[i * 3 + 1] += velocities[i * 3 + 1] * dt;
        positions[i * 3 + 2] += velocities[i * 3 + 2] * dt;

        // Apply reflective boundary conditions

        // Check x boundaries
        if (positions[i * 3 + 0] < 0.0) {
            positions[i * 3 + 0] = -positions[i * 3 + 0];
            velocities[i * 3 + 0] = -velocities[i * 3 + 0]; // Reverse x-velocity
        }
        if (positions[i * 3 + 0] >= domain[0]) {
            positions[i * 3 + 0] = 2 * domain[0] - positions[i * 3 + 0];
            velocities[i * 3 + 0] = -velocities[i * 3 + 0]; // Reverse x-velocity
        }

        // Check y boundaries
        if (positions[i * 3 + 1] < 0.0) {
            positions[i * 3 + 1] = -positions[i * 3 + 1];
            velocities[i * 3 + 1] = -velocities[i * 3 + 1]; // Reverse y-velocity
        }
        if (positions[i * 3 + 1] >= domain[1]) {
            positions[i * 3 + 1] = 2 * domain[1] - positions[i * 3 + 1];
            velocities[i * 3 + 1] = -velocities[i * 3 + 1]; // Reverse y-velocity
        }

        // Check z boundaries
        if (positions[i * 3 + 2] < 0.0) {
            positions[i * 3 + 2] = -positions[i * 3 + 2];
            velocities[i * 3 + 2] = -velocities[i * 3 + 2]; // Reverse z-velocity
        }
        if (positions[i * 3 + 2] >= domain[2]) {
            positions[i * 3 + 2] = 2 * domain[2] - positions[i * 3 + 2];
            velocities[i * 3 + 2] = -velocities[i * 3 + 2]; // Reverse z-velocity
        }
    }
}



__global__ void calculateTotalEnergy(int numParticles, double* positions, double* velocities, double* masses, double* totalEnergy, double* gravity) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    __shared__ double sharedEnergy[256]; // Assuming blockSize is 256
    sharedEnergy[threadIdx.x] = 0.0;

    // Calculate kinetic energy
    if (i < numParticles) {
        double kineticEnergy = 0.5 * masses[i] * (pow(velocities[i * 3 + 0], 2) + pow(velocities[i * 3 + 1], 2) + pow(velocities[i * 3 + 2], 2));
        sharedEnergy[threadIdx.x] += kineticEnergy;
    }

    __syncthreads();

    // Calculate potential energy
    if (i < numParticles) {
        double x = positions[i * 3 + 0];
        double y = positions[i * 3 + 1];
        double z = positions[i * 3 + 2];
        double potentialEnergy = masses[i] * (gravity[0] * x + gravity[1] * y + gravity[2] * z) * (-1);
        sharedEnergy[threadIdx.x] += potentialEnergy;
    }

    __syncthreads();

    // Reduce shared memory to get total energy for this block
    if (threadIdx.x == 0) {
        double blockEnergy = 0.0;
        for (int k = 0; k < blockDim.x; ++k) {
            blockEnergy += sharedEnergy[k];
        }
        atomicAddDouble(totalEnergy, blockEnergy);
    }
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

__global__ void neighbourlist(int numParticles, double* positions, Cell* cells, ParticleNode* nodes, double* cellSize, double* domain, int step, int num_steps ) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < numParticles) {
        // Reflect particles that go out of the domain back into the domain
        for (int dim = 0; dim < 3; ++dim) {
            if (positions[i * 3 + dim] < 0) {
                positions[i * 3 + dim] = -positions[i * 3 + dim];
            } else if (positions[i * 3 + dim] >= domain[dim]) {
                positions[i * 3 + dim] = 2 * domain[dim] - positions[i * 3 + dim];
            }
        }

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
    // Synchronize all threads to ensure all particles are inserted before printing cells
    __syncthreads();

    // Only one thread per block prints cells with particles for the first and last steps
    if (((step == 0) || (step == num_steps)) && threadIdx.x == 0) {
    if (step == 0)
        printf("At start particle distribution in cell is as:\n");
    else
        printf("At end particle distribution in cell is as:\n");
    for (int c = 0; c < (int)(domain[0] / cellSize[0]) * (int)(domain[1] / cellSize[1]) * (int)(domain[2] / cellSize[2]); ++c) {
        if (cells[c].head != -1) {
            printf("Cell (%d, %d, %d) contains particles:", c % (int)(domain[0] / cellSize[0]), (c / (int)(domain[0] / cellSize[0])) % (int)(domain[1] / cellSize[1]), c / ((int)(domain[0] / cellSize[0]) * (int)(domain[1] / cellSize[1])));
            int particleIdx = cells[c].head;
            while (particleIdx != -1) {
                printf(" %d", particleIdx);
                particleIdx = nodes[particleIdx].next;
            }
            printf("\n");
        }
    }
}
}

__global__ void calculateForcesNeighbour(int numParticles, double* positions, double* velocities, double* forces, double* masses, double g, double K, double dampingFactor, Cell* cells, ParticleNode* nodes, double* cellSize, double* domain, double particleRadius) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < numParticles) {
        // Initialize force to zero
        double force[3] = {0.0, 0.0, 0.0};

        // Apply gravitational force (mg) in the negative y-direction
        force[1] = -masses[i] * g;

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
                                // Calculate the distance vector between particles i and j
                                double xij[3] = {
                                    positions[particleIdx * 3 + 0] - positions[i * 3 + 0],
                                    positions[particleIdx * 3 + 1] - positions[i * 3 + 1],
                                    positions[particleIdx * 3 + 2] - positions[i * 3 + 2]
                                };

                                // Calculate the distance magnitude
                                double distance = sqrt(xij[0] * xij[0] + xij[1] * xij[1] + xij[2] * xij[2]);

                                // Check if particles are overlapping
                                    // Calculate the normalized distance vector
                                    double xij_norm[3] = {
                                        xij[0] / distance,
                                        xij[1] / distance,
                                        xij[2] / distance
                                    };

                                    // Calculate the relative velocity between particles i and j
                                    double vij[3] = {
                                        velocities[particleIdx * 3 + 0] - velocities[i * 3 + 0],
                                        velocities[particleIdx * 3 + 1] - velocities[i * 3 + 1],
                                        velocities[particleIdx * 3 + 2] - velocities[i * 3 + 2]
                                    };

                                    // Calculate the spring force component
                                    double overlap = 2 * particleRadius - distance; // Overlap is positive if the spring is compressed
                                    double springForce = K * overlap * overlap;  // Spring force is proportional to the square of the overlap

                                    // Calculate the dashpot force component
                                    double vij_dot_xij = vij[0] * xij_norm[0] + vij[1] * xij_norm[1] + vij[2] * xij_norm[2];  // Projection of vij on xij
                                    double dashpotForce = -dampingFactor * vij_dot_xij;  // Dashpot force is proportional to the velocity along xij

                                    // Calculate the total force between particles i and j
                                    double FSD[3] = {
                                        (springForce + dashpotForce) * xij_norm[0],
                                        (springForce + dashpotForce) * xij_norm[1],
                                        (springForce + dashpotForce) * xij_norm[2]
                                    };

                                    // Add the force to particle i
                                    force[0] += FSD[0];
                                    force[1] += FSD[1];
                                    force[2] += FSD[2];
                            }
                            particleIdx = nodes[particleIdx].next;
                        }
                    }
                }
            }
        }

        // Wall interactions using spring-dashpot model

        // Check x boundaries
        if (positions[i * 3 + 0] <= particleRadius) {
            double springForce = K * (particleRadius - positions[i * 3 + 0]) * (particleRadius - positions[i * 3 + 0]);
            double dashpotForce = -dampingFactor * velocities[i * 3 + 0];
            force[0] += springForce + dashpotForce;
        }
        if (positions[i * 3 + 0] >= domain[0] - particleRadius) {
            double springForce = K * (positions[i * 3 + 0] - (domain[0] - particleRadius)) * (positions[i * 3 + 0] - (domain[0] - particleRadius));
            double dashpotForce = dampingFactor * velocities[i * 3 + 0];
            force[0] -= springForce + dashpotForce;
        }

        // Check y boundaries
        if (positions[i * 3 + 1] <= particleRadius) {
            double springForce = K * (particleRadius - positions[i * 3 + 1]) * (particleRadius - positions[i * 3 + 1]);
            double dashpotForce = -dampingFactor * velocities[i * 3 + 1];
            force[1] += springForce + dashpotForce;
        }
        if (positions[i * 3 + 1] >= domain[1] - particleRadius) {
            double springForce = K * (positions[i * 3 + 1] - (domain[1] - particleRadius)) * (positions[i * 3 + 1] - (domain[1] - particleRadius));
            double dashpotForce = dampingFactor * velocities[i * 3 + 1];
            force[1] -= springForce + dashpotForce;
        }

        // Check z boundaries
        if (positions[i * 3 + 2] <= particleRadius) {
            double springForce = K * (particleRadius - positions[i * 3 + 2]) * (particleRadius - positions[i * 3 + 2]);
            double dashpotForce = -dampingFactor * velocities[i * 3 + 2];
            force[2] += springForce + dashpotForce;
        }
        if (positions[i * 3 + 2] >= domain[2] - particleRadius) {
            double springForce = K * (positions[i * 3 + 2] - (domain[2] - particleRadius)) * (positions[i * 3 + 2] - (domain[2] - particleRadius));
            double dashpotForce = dampingFactor * velocities[i * 3 + 2];
            force[2] -= springForce + dashpotForce;
        }

        // Store the computed force for particle i
        forces[i * 3 + 0] = force[0];
        forces[i * 3 + 1] = force[1];
        forces[i * 3 + 2] = force[2];
    }
}


void initializeParticles(int numParticles, double* positions, double* velocities, double* masses, int gridSize, double spacing, double mass, double* domain, double radius) {
    std::mt19937 gen(42);  // Fixed seed for reproducibility
    std::normal_distribution<> d(0, 1);  // Normal distribution with mean 0 and stddev 1

    int count = 0;
    double offset = spacing / 2.0;
    for (int x = 0; x < gridSize; ++x) {
        for (int y = gridSize - 1; y >= 0; --y) {
            for (int z = 0; z < gridSize; ++z) {
                if (count < numParticles) {
                    positions[count * 3 + 0] = x * spacing + offset;
                    positions[count * 3 + 1] = y * spacing + offset;
                    positions[count * 3 + 2] = z * spacing + offset;

                    // Ensure particles are within the domain boundaries considering the radius
                    positions[count * 3 + 0] = fmod(positions[count * 3 + 0], domain[0] - 2 * radius) + radius;
                    positions[count * 3 + 1] = fmod(positions[count * 3 + 1], domain[1] - 2 * radius) + radius;
                    positions[count * 3 + 2] = fmod(positions[count * 3 + 2], domain[2] - 2 * radius) + radius;

                    velocities[count * 3 + 0] = d(gen);
                    velocities[count * 3 + 1] = d(gen);
                    velocities[count * 3 + 2] = d(gen);

                    masses[count] = mass;
                    ++count;
                }
            }
        }
    }
}

void writeBoxVertices(std::ofstream &vtkFile, const double* domain) {
    vtkFile << std::fixed << std::setprecision(6);
    vtkFile << 0.0 << " " << 0.0 << " " << 0.0 << "\n";
    vtkFile << domain[0] << " " << 0.0 << " " << 0.0 << "\n";
    vtkFile << domain[0] << " " << domain[1] << " " << 0.0 << "\n";
    vtkFile << 0.0 << " " << domain[1] << " " << 0.0 << "\n";
    vtkFile << 0.0 << " " << 0.0 << " " << domain[2] << "\n";
    vtkFile << domain[0] << " " << 0.0 << " " << domain[2] << "\n";
    vtkFile << domain[0] << " " << domain[1] << " " << domain[2] << "\n";
    vtkFile << 0.0 << " " << domain[1] << " " << domain[2] << "\n";
}

void writeBoxEdges(std::ofstream &vtkFile) {
    vtkFile << "LINES 12 36\n";
    vtkFile << "2 0 1\n2 1 2\n2 2 3\n2 3 0\n";
    vtkFile << "2 4 5\n2 5 6\n2 6 7\n2 7 4\n";
    vtkFile << "2 0 4\n2 1 5\n2 2 6\n2 3 7\n";
}


int main() {
    int numParticles;
    double g, dt, time, mass,K, dampingFactor,radius;
    double domain[3],cellSize[3];
    Cell* d_cells;
    ParticleNode* d_nodes;

    std::cout << "Enter the number of particles: ";
    std::cin >> numParticles;


    // Allocate unified memory
    double* d_positions;
    double* d_velocities;
    double* d_forces;
    double* d_masses;
    double* d_domain;
    double* d_totalEnergy;
    double* d_cellSize;
    int* d_cellIndices;

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

    cudaMallocManaged(&d_positions, numParticles * 3 * sizeof(double));
    cudaMallocManaged(&d_velocities, numParticles * 3 * sizeof(double));
    cudaMallocManaged(&d_forces, numParticles * 3 * sizeof(double));
    cudaMallocManaged(&d_masses, numParticles * sizeof(double));
    cudaMallocManaged(&d_totalEnergy, sizeof(double));
    cudaMallocManaged(&d_domain, 3 * sizeof(double));
    cudaMallocManaged(&d_cells, numCells * sizeof(Cell));
    cudaMallocManaged(&d_nodes, numParticles * sizeof(ParticleNode));
    cudaMallocManaged(&d_cellSize, 3 * sizeof(double));
    cudaMallocManaged(&d_cellIndices, numParticles * sizeof(int));

    int gridSize = std::ceil(std::pow(numParticles, 1.0 / 3.0));
    double spacing = 1.0;

    std::cout << "Enter the spring constant: ";
    std::cin >> K;
    std::cout << "Enter the damping constant: ";
    std::cin >> dampingFactor;

    cudaMemcpy(d_domain, domain, 3 * sizeof(double), cudaMemcpyHostToDevice);

    initializeParticles(numParticles, d_positions, d_velocities, d_masses, gridSize, spacing, mass, domain, radius);
    initializeCells<<<(numCells + numParticles + 255) / 256, 256>>>(numCells, d_cells, d_nodes, numParticles);

    std::cout << "Enter gravity : ";
    std::cin >> g;
    std::cout << "Enter size of time steps: ";
    std::cin >> dt;
    std::cout << "Enter the end time for simulation: ";
    std::cin >> time;

    cudaMemcpy(d_domain, domain, 3 * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_cellSize, cellSize, 3 * sizeof(double), cudaMemcpyHostToDevice);

    int numSteps = time / dt;

    // Open initial VTK file
    std::ofstream vtkFile("particles_0.vtk");
    vtkFile << "# vtk DataFile Version 3.0\n";
    vtkFile << "Particle Simulation\n";
    vtkFile << "ASCII\n";
    vtkFile << "DATASET POLYDATA\n";
    vtkFile << "POINTS " << numParticles + 8 << " float\n";

    int blockSize = 256;
    int numBlocks = (numParticles + blockSize - 1) / blockSize;

    float totalTimeForComputation = 0.0f;

    // Allocate host memory for copying positions
    double* h_positions = new double[numParticles * 3];

    for (int step = 0; step <= numSteps; ++step) 
    {
        cudaEvent_t start, stop;
        float elapsedTime;
        cudaEventCreate(&start);
        cudaEventRecord(start, 0);

        // 1. Write current positions of all particles to VTK file
        cudaMemcpy(h_positions, d_positions, numParticles * 3 * sizeof(double), cudaMemcpyDeviceToHost);

        vtkFile << std::fixed << std::setprecision(6);
        for (int i = 0; i < numParticles; ++i) 
        {
            vtkFile << h_positions[i * 3 + 0] << " " << h_positions[i * 3 + 1] << " " << h_positions[i * 3 + 2] << "\n";
        }

        // Write box vertices
        writeBoxVertices(vtkFile, domain);

        // 2. initialize cells
        initializeCells<<<(numCells + numParticles + 255) / 256, 256>>>(numCells, d_cells, d_nodes, numParticles);
        cudaDeviceSynchronize();

        // 3. Calculate forces
        if (step == 0) 
        {   
            calculateForces<<<numBlocks, blockSize>>>(numParticles, d_positions,d_velocities, d_forces, d_masses, g, K, dampingFactor, radius, d_domain);
            cudaDeviceSynchronize();
        }
        else
        {
            calculateForcesNeighbour<<<numBlocks, blockSize>>>(numParticles, d_positions,d_velocities, d_forces, d_masses, g, K, dampingFactor, d_cells, d_nodes, d_cellSize, d_domain, radius);
            cudaDeviceSynchronize();
        } 

        // 4. First integration step
        updateParticles<<<numBlocks, blockSize>>>(numParticles, d_positions, d_velocities, d_forces, d_masses, dt, d_domain);
        cudaDeviceSynchronize();
        
        // 5. calculate updated cell for each particle
        if (step == 0 && step == numSteps){
            printf("Launching kernel for step: %d\n", step);
        }
        neighbourlist<<<(numParticles + 255) / 256, 256>>>(numParticles, d_positions, d_cells, d_nodes, d_cellSize, d_domain,step, numSteps);
        cudaDeviceSynchronize();

        
        

        // 8. Calculate total energy
        cudaMemset(d_totalEnergy, 0, sizeof(double));
        calculateTotalEnergy<<<numBlocks, blockSize>>>(numParticles, d_positions, d_velocities, d_masses, d_totalEnergy, d_domain);
        cudaDeviceSynchronize();


        // Write box edges
        writeBoxEdges(vtkFile);

        cudaEventCreate(&stop);
        cudaEventRecord(stop, 0);
        cudaEventSynchronize(stop);

        cudaEventElapsedTime(&elapsedTime, start, stop);
        totalTimeForComputation += elapsedTime;

        double totalEnergy;
        cudaMemcpy(&totalEnergy, d_totalEnergy, sizeof(double), cudaMemcpyDeviceToHost);

        if (step % 5 == 0) {
            // Write total energy to file
            std::ofstream energyFile("energy.txt", std::ofstream::app);
            energyFile << "Step: " << step << " Total Energy: " << totalEnergy << "\n";
            energyFile.close();

            // Close and open the VTK file for the next step
            vtkFile.close();
            vtkFile.open("particles_" + std::to_string(step + 1) + ".vtk");
            vtkFile << "# vtk DataFile Version 3.0\n";
            vtkFile << "Particle Simulation\n";
            vtkFile << "ASCII\n";
            vtkFile << "DATASET POLYDATA\n";
            vtkFile << "POINTS " << numParticles + 8 << " float\n";

            std::ofstream outFile("time.txt", std::ofstream::app); 
            outFile << "Time: " << elapsedTime << "ms" << std::endl;
            outFile.close();
        }
    }

    printf("Total time for computation is : %f ms \n", totalTimeForComputation);

    // Close final VTK file
    vtkFile.close();

    // Free unified memory
    cudaFree(d_positions);
    cudaFree(d_velocities);
    cudaFree(d_forces);
    cudaFree(d_masses);
    cudaFree(d_domain);
    cudaFree(d_totalEnergy);
    cudaFree(d_cells);
    cudaFree(d_nodes);
    cudaFree(d_cellIndices);

    //delete[] h_positions;

    return 0;
}
