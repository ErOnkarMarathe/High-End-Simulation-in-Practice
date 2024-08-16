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
__device__ double distance(const double* pos1, const double* pos2) {
    return sqrt(pow(pos2[0] - pos1[0], 2) + pow(pos2[1] - pos1[1], 2) + pow(pos2[2] - pos1[2], 2));
}

__global__ void calculateForces(int numParticles, double* positions, double* forces, double sigma, double epsilon) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < numParticles) {
        double force[3] = {0.0, 0.0, 0.0};
        for (int j = 0; j < numParticles; ++j) {
            if (i != j) {
                double r = distance(&positions[i * 3], &positions[j * 3]);
                if (r != 0) {
                    double coeff = 24 * epsilon / r * (2 * pow(sigma / r, 12) - pow(sigma / r, 6));
                    force[0] += coeff * (positions[i * 3 + 0] - positions[j * 3 + 0]) / r;
                    force[1] += coeff * (positions[i * 3 + 1] - positions[j * 3 + 1]) / r;
                    force[2] += coeff * (positions[i * 3 + 2] - positions[j * 3 + 2]) / r;
                }
            }
        }
        forces[i * 3 + 0] = force[0];
        forces[i * 3 + 1] = force[1];
        forces[i * 3 + 2] = force[2];
    }
}

__global__ void updateParticles(int numParticles, double* positions, double* velocities, double* forces, double* masses, double dt) {
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

__global__ void calculateTotalEnergy(int numParticles, double* positions, double* velocities, double* masses, double sigma, double epsilon, double* totalEnergy) {
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
            double r = distance(&positions[i * 3], &positions[j * 3]);
            if (r != 0) {
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

void initializeParticles(int numParticles, double* positions, double* velocities, double* masses, int gridSize, double spacing, double mass) {
    //std::random_device rd;
    std::mt19937 gen(42);
    std::normal_distribution<> d(0, 1);

    int count = 0;
    for (int x = 0; x < gridSize; ++x) {
        for (int y = 0; y < gridSize; ++y) {
            for (int z = 0; z < gridSize; ++z) {
                if (count < numParticles) {
                    positions[count * 3 + 0] = x * spacing;
                    positions[count * 3 + 1] = y * spacing;
                    positions[count * 3 + 2] = z * spacing;

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

int main() {
    int numParticles;
    double sigma, epsilon, dt, time, mass;

    std::cout << "Enter the number of particles: ";
    std::cin >> numParticles;

    // Allocate unified memory
    double* d_positions;
    double* d_velocities;
    double* d_forces;
    double* d_masses;
    double* d_totalEnergy;

    cudaMallocManaged(&d_positions, numParticles * 3 * sizeof(double));
    cudaMallocManaged(&d_velocities, numParticles * 3 * sizeof(double));
    cudaMallocManaged(&d_forces, numParticles * 3 * sizeof(double));
    cudaMallocManaged(&d_masses, numParticles * sizeof(double));
    cudaMallocManaged(&d_totalEnergy, sizeof(double));

    int gridSize = std::ceil(std::pow(numParticles, 1.0 / 3.0));
    double spacing = 1.0;

    std::cout << "Enter the mass of each particle: ";
    std::cin >> mass;

    initializeParticles(numParticles, d_positions, d_velocities, d_masses, gridSize, spacing, mass);

    std::cout << "Enter Lennard-Jones constant sigma: ";
    std::cin >> sigma;
    std::cout << "Enter Lennard-Jones constant epsilon: ";
    std::cin >> epsilon;
    std::cout << "Enter size of time steps: ";
    std::cin >> dt;
    std::cout << "Enter the end time for simulation: ";
    std::cin >> time;

    int num_steps = time / dt;
    // Open initial VTK file
    std::ofstream vtkFile("particles_0.vtk");
    vtkFile << "# vtk DataFile Version 3.0\n";
    vtkFile << "Particle Simulation\n";
    vtkFile << "ASCII\n";
    vtkFile << "DATASET POLYDATA\n";
    vtkFile << "POINTS " << numParticles << " float\n";

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

        // 2. Calculate forces
        if (step == 0) 
        {
        calculateForces<<<numBlocks, blockSize>>>(numParticles, d_positions, d_forces, sigma, epsilon);
        cudaDeviceSynchronize();
        }

        // 3. First integration step
        updateParticles<<<numBlocks, blockSize>>>(numParticles, d_positions, d_velocities, d_forces, d_masses, dt);
        cudaDeviceSynchronize();

        // 4. Calculate new forces
        calculateForces<<<numBlocks, blockSize>>>(numParticles, d_positions, d_forces, sigma, epsilon);
        cudaDeviceSynchronize();

        // 5. Finalize velocities
        finalizeVelocities<<<numBlocks, blockSize>>>(numParticles, d_velocities, d_forces, d_masses, dt);
        cudaDeviceSynchronize();

        // 6. Calculate total energy
        cudaMemset(d_totalEnergy, 0, sizeof(double));
        calculateTotalEnergy<<<numBlocks, blockSize>>>(numParticles, d_positions, d_velocities, d_masses, sigma, epsilon, d_totalEnergy);
        cudaDeviceSynchronize();

        cudaEventCreate(&stop);
        cudaEventRecord(stop,0);
        cudaEventSynchronize(stop);

        cudaEventElapsedTime(&elapsedTime, start,stop);
        //printf("Elapsed time : %f ms\n" ,elapsedTime);
        TotalTimeforComputation +=elapsedTime; 

        double totalEnergy;
        cudaMemcpy(&totalEnergy, d_totalEnergy, sizeof(double), cudaMemcpyDeviceToHost);

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
        vtkFile << "POINTS " << numParticles << " float\n";

        std::ofstream outFile("time.txt", std::ofstream::app); 
        outFile << "Time: " << elapsedTime << "ms" << std::endl;
        outFile.close();
    }

    printf("Total time for computaion is : %fms \n", TotalTimeforComputation);


    // Close final VTK file
    vtkFile.close();

    // Free unified memory
    cudaFree(d_positions);
    cudaFree(d_velocities);
    cudaFree(d_forces);
    cudaFree(d_masses);
    cudaFree(d_totalEnergy);

    return 0;
}
