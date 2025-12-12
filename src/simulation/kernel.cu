#include <cuda_runtime.h>
#include <iostream>

// ... (Garde le __global__ void simulateAndGates tel quel au début) ...
__global__ void simulateAndGates(int* entreeA, int* entreeB, int* sorties, int nombreDePortes) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < nombreDePortes) {
        sorties[i] = entreeA[i] & entreeB[i];
    }
}

// --- MODIFICATION ICI ---
// On change la signature pour accepter un pointeur vers le tableau de résultats
extern "C" void launchGPUCalculation(int* host_results, int N) {
    size_t size = N * sizeof(int);

    // 1. Allocation Host (On utilise les tableaux temporaires pour les entrées seulement)
    int *h_A, *h_B;
    cudaMallocHost(&h_A, size);
    cudaMallocHost(&h_B, size);

    // 2. Remplissage (On simule une horloge : 0 1 0 1 0 1...)
    for (int i = 0; i < N; i++) {
        h_A[i] = 1;      
        h_B[i] = i % 2;  // Cela va créer une oscillation parfaite
    }

    // 3. Allocation GPU
    int *d_A, *d_B, *d_Out;
    cudaMalloc(&d_A, size);
    cudaMalloc(&d_B, size);
    cudaMalloc(&d_Out, size);

    // 4. Copie CPU -> GPU
    cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);

    // 5. Run Kernel
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
    simulateAndGates<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_Out, N);
    cudaDeviceSynchronize();

    // 6. RÉCUPÉRATION (On copie directement dans le tableau fourni par le Main)
    cudaMemcpy(host_results, d_Out, size, cudaMemcpyDeviceToHost);

    // Nettoyage
    cudaFree(d_A); cudaFree(d_B); cudaFree(d_Out);
    cudaFreeHost(h_A); cudaFreeHost(h_B);
}