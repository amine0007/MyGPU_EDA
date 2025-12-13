#include <cuda_runtime.h>
#include <iostream>
#include "Netlist.hpp"

// --- KERNEL 1 : CALCUL LOGIQUE ---
// Pour les portes classiques, on calcule.
// Pour la DFF, on sort juste la valeur stockée (état précédent).
__global__ void simulateLogic(Gate* gates, int* signals, int* states, int nbGates) {
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
            case GATE_DFF: result = states[idx]; break; // La sortie est la mémoire interne
        }
        signals[g.output_id] = result;
    }
}

// --- KERNEL 2 : MISE À JOUR MÉMOIRE (Front d'horloge) ---
// On regarde l'entrée de la DFF et on la stocke pour le prochain tour.
__global__ void updateDFFs(Gate* gates, int* signals, int* states, int nbGates) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < nbGates) {
        Gate g = gates[idx];
        if (g.type == GATE_DFF) {
            // On capture l'entrée et on la met en mémoire
            states[idx] = signals[g.input1_id];
        }
    }
}

extern "C" void launchGPUCalculation(Gate* host_gates, int numGates, 
                                     int* host_probes, int numProbes, 
                                     int* host_waveform_out, int nb_cycles) {
    
    size_t gateSize = numGates * sizeof(Gate);
    int numSignals = 1024; 
    
    int* d_signals;
    int* d_states; // Tableau mémoire
    Gate* d_gates;
    
    cudaMalloc(&d_signals, numSignals * sizeof(int));
    cudaMalloc(&d_states, numGates * sizeof(int)); // Un état par porte
    cudaMalloc(&d_gates, gateSize);
    
    cudaMemcpy(d_gates, host_gates, gateSize, cudaMemcpyHostToDevice);
    cudaMemset(d_signals, 0, numSignals * sizeof(int));
    cudaMemset(d_states, 0, numGates * sizeof(int)); // Mémoire vide au début

    // BOUCLE TEMPORELLE
    for (int t = 0; t < nb_cycles; t++) {
        
        // 1. Stimuli (Horloge sur le fil 0)
        int inputs[1];
        inputs[0] = t % 2; // Horloge rapide 0, 1, 0, 1...
        cudaMemcpy(d_signals, inputs, 1 * sizeof(int), cudaMemcpyHostToDevice);

        int threads = 256;
        int blocks = (numGates + threads - 1) / threads;

        // 2. PHASE 1 : Calcul des sorties
        simulateLogic<<<blocks, threads>>>(d_gates, d_signals, d_states, numGates);
        cudaDeviceSynchronize();

        // 3. PHASE 2 : Mise à jour des mémoires (Sauvegarde pour t+1)
        updateDFFs<<<blocks, threads>>>(d_gates, d_signals, d_states, numGates);
        cudaDeviceSynchronize();

        // 4. Extraction Sondes
        for (int p = 0; p < numProbes; p++) {
            int val_sonde;
            int id_fil = host_probes[p];
            cudaMemcpy(&val_sonde, &d_signals[id_fil], sizeof(int), cudaMemcpyDeviceToHost);
            host_waveform_out[t * numProbes + p] = val_sonde;
        }
    }

    cudaFree(d_signals);
    cudaFree(d_states);
    cudaFree(d_gates);
}