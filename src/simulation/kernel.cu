#include <cuda_runtime.h>
#include <iostream>

// --- LE KERNEL (Ce qui tourne sur la carte graphique) ---
// Chaque "thread" GPU va s'occuper d'une seule porte logique
__global__ void simulateAndGates(int* entreeA, int* entreeB, int* sorties, int nombreDePortes) {
    // Calcul de l'index unique du thread (la "carte d'identité" du thread)
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    // Si l'index est valide (pour ne pas dépasser le tableau)
    if (i < nombreDePortes) {
        // La porte logique : A AND B
        sorties[i] = entreeA[i] & entreeB[i];
    }
}

// --- LE WRAPPER (La fonction C++ qui lance le GPU) ---
extern "C" void launchGPUCalculation() {
    int N = 1000000; // 1 Million de portes !
    size_t size = N * sizeof(int);

    // 1. Allocation de la mémoire CPU (Host)
    int *h_A, *h_B, *h_Out;
    cudaMallocHost(&h_A, size);
    cudaMallocHost(&h_B, size);
    cudaMallocHost(&h_Out, size);

    // 2. Remplissage des données (Simulation de signaux aléatoires)
    for (int i = 0; i < N; i++) {
        h_A[i] = 1;      // Tout à 1
        h_B[i] = i % 2;  // 0, 1, 0, 1...
    }

    // 3. Allocation de la mémoire GPU (Device)
    int *d_A, *d_B, *d_Out;
    cudaMalloc(&d_A, size);
    cudaMalloc(&d_B, size);
    cudaMalloc(&d_Out, size);

    // 4. Copie CPU -> GPU
    cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);

    // 5. LANCEMENT DU KERNEL (256 threads par bloc)
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
    
    std::cout << "Lancement de la simulation sur GPU pour " << N << " portes..." << std::endl;
    simulateAndGates<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_Out, N);
    
    // Attendre que le GPU finisse
    cudaDeviceSynchronize();

    // 6. Copie GPU -> CPU (Récupérer les résultats)
    cudaMemcpy(h_Out, d_Out, size, cudaMemcpyDeviceToHost);

    // 7. Vérification des 5 premiers résultats
    std::cout << "Resultats des 5 premieres portes :" << std::endl;
    for(int i=0; i<5; i++) {
        std::cout << "Porte " << i << ": " << h_A[i] << " AND " << h_B[i] << " = " << h_Out[i] << std::endl;
    }

    // Nettoyage
    cudaFree(d_A); cudaFree(d_B); cudaFree(d_Out);
    cudaFreeHost(h_A); cudaFreeHost(h_B); cudaFreeHost(h_Out);
}