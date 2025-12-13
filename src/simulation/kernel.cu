#include <cuda_runtime.h>
#include <iostream>
#include "Netlist.hpp"

// ... (Garde le __global__ void simulateCircuit EXACTEMENT comme avant) ...
__global__ void simulateCircuit(Gate* gates, int* signals, int nbGates) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < nbGates) {
        Gate g = gates[idx];
        int val1 = signals[g.input1_id];
        int val2 = signals[g.input2_id];
        int result = 0;
        switch(g.type) {
            case GATE_AND: result = val1 & val2; break;
            case GATE_OR:  result = val1 | val2; break;
            case GATE_XOR: result = val1 ^ val2; break;
            case GATE_NOT: result = !val1;       break;
        }
        signals[g.output_id] = result;
    }
}

// --- MODIFICATION MAJEURE ICI ---
// On accepte désormais le tableau de portes (circuit) venant du CPU (main.cpp)
extern "C" void launchGPUCalculation(Gate* host_gates, int numGates, int* host_waveform_out, int nb_cycles) {
    
    // 1. Calcul de la taille mémoire
    size_t gateSize = numGates * sizeof(Gate);
    
    // 2. Préparation Mémoire GPU
    // On prévoit large pour les fils (ex: 1024 fils possibles)
    int numSignals = 1024; 
    int* d_signals;
    Gate* d_gates;
    
    cudaMalloc(&d_signals, numSignals * sizeof(int));
    cudaMalloc(&d_gates, gateSize);
    
    // 3. Copie du Circuit (CPU -> GPU)
    // C'est ici qu'on envoie ce qu'on a lu dans le fichier texte
    cudaMemcpy(d_gates, host_gates, gateSize, cudaMemcpyHostToDevice);
    
    // Mise à zéro des signaux
    cudaMemset(d_signals, 0, numSignals * sizeof(int));

    // 4. BOUCLE TEMPORELLE
    for (int t = 0; t < nb_cycles; t++) {
        
        // Génération des stimuli (Toujours hardcodé pour l'instant : Horloges sur 0 et 1)
        int inputs[2];
        inputs[0] = t % 2;       
        inputs[1] = (t / 3) % 2; // Notre chaos désynchronisé

        cudaMemcpy(d_signals, inputs, 2 * sizeof(int), cudaMemcpyHostToDevice);

        // Lancement du Kernel
        int threads = 256;
        int blocks = (numGates + threads - 1) / threads;
        simulateCircuit<<<blocks, threads>>>(d_gates, d_signals, numGates);
        cudaDeviceSynchronize();

        // On récupère le fil 4 (Notre sortie XOR définie dans le fichier texte)
        int val_sortie;
        cudaMemcpy(&val_sortie, &d_signals[4], sizeof(int), cudaMemcpyDeviceToHost);
        host_waveform_out[t] = val_sortie;
    }

    cudaFree(d_signals);
    cudaFree(d_gates);
}