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

// Helper function to calculate the distance between two particles in 3D
__device__ double distance(const double* pos1, const double* pos2, const double* domain) {
    double dx = pos2[0] - pos1[0];
    double dy = pos2[1] - pos1[1];
    double dz = pos2[2] - pos1[2];

    // Apply periodic boundary conditions
    if (dx > domain[0] / 2.0) dx -= domain[0];
    else if (dx < -domain[0] / 2.0) dx += domain[0];
    
    if (dy > domain[1] / 2.0) dy -= domain[1];
    else if (dy < -domain[1] / 2.0) dy += domain[1];
    
    if (dz > domain[2] / 2.0) dz -= domain[2];
    else if (dz < -domain[2] / 2.0) dz += domain[2];

    // Compute the minimum distance using the adjusted coordinates
    return sqrt(dx * dx + dy * dy + dz * dz);
}

__global__ void calculateForces(int numParticles, double* positions, double* forces, double sigma, double epsilon, double* domain, double cutoff) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < numParticles) {
        double force[3] = {0.0, 0.0, 0.0};
        for (int j = 0; j < numParticles; ++j) {
            if (i != j) {
                double r = distance(&positions[i * 3], &positions[j * 3], domain); // Pass domain to distance function
                if (r != 0 && r < cutoff) {
                    double coeff = 24 * epsilon / r * (2 * pow(sigma / r, 12) - pow(sigma / r, 6));
                    double dx = positions[i * 3 + 0] - positions[j * 3 + 0];
                    double dy = positions[i * 3 + 1] - positions[j * 3 + 1];
                    double dz = positions[i * 3 + 2] - positions[j * 3 + 2];

                    // Apply minimum image convention
                    dx -= domain[0] * round(dx / domain[0]);
                    dy -= domain[1] * round(dy / domain[1]);
                    dz -= domain[2] * round(dz / domain[2]);

                    force[0] += coeff * dx / r;
                    force[1] += coeff * dy / r;
                    force[2] += coeff * dz / r;
                }
            }
        }
        forces[i * 3 + 0] = force[0];
        forces[i * 3 + 1] = force[1];
        forces[i * 3 + 2] = force[2];
    }
}

__global__ void updateParticles(int numParticles, double* positions, double* velocities, double* forces, double* masses, double dt, double* domain) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < numParticles) {
        double acc[3];
        acc[0] = forces[i * 3 + 0] / masses[i];
        acc[1] = forces[i * 3 + 1] / masses[i];
        acc[2] = forces[i * 3 + 2] / masses[i];

        positions[i * 3 + 0] += velocities[i * 3 + 0] * dt + 0.5 * acc[0] * dt * dt;
        positions[i * 3 + 1] += velocities[i * 3 + 1] * dt + 0.5 * acc[1] * dt * dt;
        positions[i * 3 + 2] += velocities[i * 3 + 2] * dt + 0.5 * acc[2] * dt * dt;

        velocities[i * 3 + 0] += 0.5 * acc[0] * dt;
        velocities[i * 3 + 1] += 0.5 * acc[1] * dt;
        velocities[i * 3 + 2] += 0.5 * acc[2] * dt;

        // Apply periodic boundary conditions
        positions[i * 3 + 0] = fmod(positions[i * 3 + 0] + domain[0], domain[0]);
        positions[i * 3 + 1] = fmod(positions[i * 3 + 1] + domain[1], domain[1]);
        positions[i * 3 + 2] = fmod(positions[i * 3 + 2] + domain[2], domain[2]);

        if (positions[i * 3 + 0] < 0) positions[i * 3 + 0] += domain[0];
        if (positions[i * 3 + 1] < 0) positions[i * 3 + 1] += domain[1];
        if (positions[i * 3 + 2] < 0) positions[i * 3 + 2] += domain[2];
    }
}

__global__ void finalizeVelocities(int numParticles, double* velocities, double* forces, double* masses, double dt) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < numParticles) {
        double acc[3];
        acc[0] = forces[i * 3 + 0] / masses[i];
        acc[1] = forces[i * 3 + 1] / masses[i];
        acc[2] = forces[i * 3 + 2] / masses[i];

        velocities[i * 3 + 0] += 0.5 * acc[0] * dt;
        velocities[i * 3 + 1] += 0.5 * acc[1] * dt;
        velocities[i * 3 + 2] += 0.5 * acc[2] * dt;
    }
}

__global__ void calculateTotalEnergy(int numParticles, double* positions, double* velocities, double* masses, double sigma, double epsilon, double* totalEnergy, double* domain, double cutoff) {
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
        for (int j = i + 1; j < numParticles; ++j) {
            double r = distance(&positions[i * 3], &positions[j * 3],domain);
            if (r <= cutoff) {
                double potentialEnergy = 4 * epsilon * (pow(sigma / r, 12) - pow(sigma / r, 6));
                sharedEnergy[threadIdx.x] += potentialEnergy;
            }
        }
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


__global__ void neighbourlist(int numParticles, double* positions, Cell* cells, ParticleNode* nodes, double* cellSize, double* domain, int step, int num_steps) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < numParticles) {
        // Compute the cell index for the particle
        int cellX = (int)(positions[i * 3 + 0]  / cellSize[0]);
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

__global__ void calculateForcesNeighbour(int numParticles, double* positions, double* forces, double sigma, double epsilon, double* domain, double cutoff, ParticleNode* nodes, double* cellSize,Cell* cells) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < numParticles) {
        double force[3] = {0.0, 0.0, 0.0};

        // Get the cell index for the particle
        int cellX = (int)(positions[i * 3 + 0]  / cellSize[0]);
        int cellY = (int)(positions[i * 3 + 1] / cellSize[1]);
        int cellZ = (int)(positions[i * 3 + 2] / cellSize[2]);

        // Ensure the particle is within the grid boundaries
        cellX = min(max(cellX, 0), (int)(domain[0] / cellSize[0]) - 1);
        cellY = min(max(cellY, 0), (int)(domain[1] / cellSize[1]) - 1);
        cellZ = min(max(cellZ, 0), (int)(domain[2] / cellSize[2]) - 1);

        int cellIndex = cellZ * (int)(domain[0] / cellSize[0]) * (int)(domain[1] / cellSize[1]) + cellY * (int)(domain[0] / cellSize[0]) + cellX;

        // Loop through particles in the same cell
        int particleIdx = cells[cellIndex].head;
        while (particleIdx != -1) {
            if (particleIdx != i) {
                double dx = positions[i * 3 + 0] - positions[particleIdx * 3 + 0];
                double dy = positions[i * 3 + 1] - positions[particleIdx * 3 + 1];
                double dz = positions[i * 3 + 2] - positions[particleIdx * 3 + 2];

                // Apply minimum image convention
                dx -= domain[0] * round(dx / domain[0]);
                dy -= domain[1] * round(dy / domain[1]);
                dz -= domain[2] * round(dz / domain[2]);

                double rSquared = dx * dx + dy * dy + dz * dz;
                double r = sqrt(rSquared);

                if (r != 0 && r < cutoff) {
                    double coeff = 24 * epsilon / r * (2 * pow(sigma / r, 12) - pow(sigma / r, 6));
                    force[0] += coeff * dx / r;
                    force[1] += coeff * dy / r;
                    force[2] += coeff * dz / r;
                }
            }
            particleIdx = nodes[particleIdx].next;
        }

        forces[i * 3 + 0] = force[0];
        forces[i * 3 + 1] = force[1];
        forces[i * 3 + 2] = force[2];
    }
}


void initializeParticles(int numParticles, double* positions, double* velocities, double* masses, int gridSize, double spacing, double mass, double* domain) {
    //std::random_device rd;
    std::mt19937 gen(42);
    std::normal_distribution<> d(0, 1);

    int count = 0;
    double offset = spacing / 2.0;
    for (int x = 0; x < gridSize; ++x) {
        for (int y = 0; y < gridSize; ++y) {
            for (int z = 0; z < gridSize; ++z) {
                if (count < numParticles) {
                    positions[count * 3 + 0] = x * spacing + offset;
                    positions[count * 3 + 1] = y * spacing + offset;
                    positions[count * 3 + 2] = z * spacing + offset;

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
    double sigma, epsilon, dt, time, mass,cutoff;
    double domain[3];
    double cellSize[3];
    Cell* d_cells;
    ParticleNode* d_nodes;

    std::cout << "Enter the number of particles: ";
    std::cin >> numParticles;
    std::cout << "Enter the mass of each particle: ";
    std::cin >> mass;

    std::cout << "Enter domain dimensions (x y z): ";
    std::cin >> domain[0] >> domain[1] >> domain[2];

    std::cout << "Enter cell dimensions (x y z): ";
    std::cin >> cellSize[0] >> cellSize[1] >> cellSize[2];
    int numCells = (domain[0] * domain[1] * domain[2]) / (cellSize[0] * cellSize[1] * cellSize[2]);

    std::cout << "Total number of cells: " << numCells << std::endl;
    
    // Allocate unified memory
    double* d_positions;
    double* d_velocities;
    double* d_forces;
    double* d_masses;
    double* d_totalEnergy;
    double* d_domain;
    double* d_cellSize;

    cudaMallocManaged(&d_positions, numParticles * 3 * sizeof(double));
    cudaMallocManaged(&d_velocities, numParticles * 3 * sizeof(double));
    cudaMallocManaged(&d_forces, numParticles * 3 * sizeof(double));
    cudaMallocManaged(&d_masses, numParticles * sizeof(double));
    cudaMallocManaged(&d_totalEnergy, sizeof(double));
    cudaMallocManaged(&d_domain, 3 * sizeof(double));
    cudaMallocManaged(&d_cells, numCells * sizeof(Cell));
    cudaMallocManaged(&d_nodes, numParticles * sizeof(ParticleNode));
    cudaMallocManaged(&d_cellSize, 3 * sizeof(double));

    int gridSize = std::ceil(std::pow(numParticles, 1.0 / 3.0));
    double spacing = 1.0;


    cudaMemcpy(d_domain, domain, 3 * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_cellSize, cellSize, 3 * sizeof(double), cudaMemcpyHostToDevice);

    initializeParticles(numParticles, d_positions, d_velocities, d_masses, gridSize, spacing, mass, domain);
    initializeCells<<<(numCells + numParticles + 255) / 256, 256>>>(numCells, d_cells, d_nodes, numParticles);
    cudaDeviceSynchronize();

    std::cout << "Enter Lennard-Jones constant sigma: ";
    std::cin >> sigma;
    std::cout << "Enter Lennard-Jones constant epsilon: ";
    std::cin >> epsilon;
    std::cout << "Enter size of time steps: ";
    std::cin >> dt;
    std::cout << "Enter the end time for simulation: ";
    std::cin >> time;
    std::cout << "Enter the cutoff distance: ";
    std::cin >> cutoff;


    int num_steps = time / dt;
    
    // Open initial VTK file
    std::ofstream vtkFile("particles_0.vtk");
    vtkFile << "# vtk DataFile Version 3.0\n";
    vtkFile << "Particle Simulation\n";
    vtkFile << "ASCII\n";
    vtkFile << "DATASET POLYDATA\n";
    vtkFile << "POINTS " << numParticles + 8 << " float\n";

    int blockSize = 256;
    int numBlocks = (numParticles + blockSize - 1) / blockSize;

    float TotalTimeforComputation = 0.0f;

    for (int step = 0; step <= num_steps; ++step) 
    {
        cudaEvent_t start, stop;
        float elapsedTime;
        cudaEventCreate(&start);
        cudaEventRecord(start,0);
        
        // 1. Write current positions of all particles to VTK file
        vtkFile << std::fixed << std::setprecision(6);
        for (int i = 0; i < numParticles; ++i) 
        {
            vtkFile << d_positions[i * 3 + 0] << " " << d_positions[i * 3 + 1] << " " << d_positions[i * 3 + 2] << "\n";
        }

        // Write box vertices
        writeBoxVertices(vtkFile, domain);


        // 2. initialize cells
        initializeCells<<<(numCells + numParticles + 255) / 256, 256>>>(numCells, d_cells, d_nodes, numParticles);
        cudaDeviceSynchronize();

        // 3. Calculate forces for time step 0
        if (step == 0) 
        {
        calculateForces<<<numBlocks, blockSize>>>(numParticles, d_positions, d_forces, sigma, epsilon, d_domain,cutoff);
        cudaDeviceSynchronize();
        }

        // 4. First integration step
        updateParticles<<<numBlocks, blockSize>>>(numParticles, d_positions, d_velocities, d_forces, d_masses, dt, d_domain);
        cudaDeviceSynchronize();

        // 5. calculate updated cell for each particle
        if (step == 0 && step == num_steps){
            printf("Launching kernel for step: %d\n", step);
        }
        neighbourlist<<<(numParticles + 255) / 256, 256>>>(numParticles, d_positions, d_cells, d_nodes, d_cellSize, d_domain,step, num_steps);
        cudaDeviceSynchronize();

        // 6. Calculate new forces using list
        //calculateForces<<<numBlocks, blockSize>>>(numParticles, d_positions, d_forces, sigma, epsilon, d_domain,cutoff);
        //cudaDeviceSynchronize();
        calculateForcesNeighbour<<<numBlocks, blockSize>>>(numParticles, d_positions, d_forces, sigma, epsilon, d_domain, cutoff, d_nodes, d_cellSize, d_cells);
        cudaDeviceSynchronize();

        // 7. Finalize velocities
        finalizeVelocities<<<numBlocks, blockSize>>>(numParticles, d_velocities, d_forces, d_masses, dt);
        cudaDeviceSynchronize();

        // 8. Calculate total energy
        cudaMemset(d_totalEnergy, 0, sizeof(double));
        calculateTotalEnergy<<<numBlocks, blockSize>>>(numParticles, d_positions, d_velocities, d_masses, sigma, epsilon, d_totalEnergy, d_domain,cutoff);
        cudaDeviceSynchronize();
        

        // Write box edges
        vtkFile << "LINES 12 36\n";
        writeBoxEdges(vtkFile);


        cudaEventCreate(&stop);
        cudaEventRecord(stop,0);
        cudaEventSynchronize(stop);

        cudaEventElapsedTime(&elapsedTime, start,stop);
        TotalTimeforComputation +=elapsedTime; 

        double totalEnergy;
        cudaMemcpy(&totalEnergy, d_totalEnergy, sizeof(double), cudaMemcpyDeviceToHost);
    
        if (step % 50 == 0) {
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

    printf("Total time for computaion is : %fms \n", TotalTimeforComputation);

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

    return 0;
}
